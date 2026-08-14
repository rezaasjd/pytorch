# Owner(s): ["module: dynamo"]
# test/dynamo/test_privateuse1_stream_handler.py
#
# Regression test for https://github.com/pytorch/pytorch/pull/192970
#
# Verifies that out-of-tree PrivateUse1 backends registered via DeviceInterface
# get correct stream handling in Dynamo tracing, addressing the root cause of
# the wait_stream(None) bug (streams::wait_stream received None instead of a
# valid stream index for backends outside the hardcoded CUDA/XPU list).
#
# The PR fixes this by iterating ``get_registered_device_interfaces()`` inside
# ``TorchInGraphFunctionVariable._get_handlers()``, so any backend that has
# registered a DeviceInterface via ``register_interface_for_device()``
# automatically gets its ``current_stream`` mapped to the device-agnostic
# ``handle_current_stream`` handler — no name-guessing, no hardcoded list.
#
# This test simulates what torch_npu does at import time (trace_rule.py
# registers "torch.npu.current_stream" as TorchInGraphFunctionVariable) using
# a fake device module ``torch.fake_device``, so the test exercises the same
# code path as a real PrivateUse1 backend without requiring NPU or any
# specific accelerator hardware beyond what ``torch.accelerator`` provides.
#
# All tests are gated on ``torch.accelerator.is_available()`` because
# SymbolicStreamState.__init__ (streams.py:272) leaves cur_stream_stack empty
# when no accelerator is present, causing handle_current_stream to graph-break
# through its except clause before the handler lookup can complete.

import sys
import types
import unittest

import torch
from torch._dynamo import trace_rules
from torch._dynamo.device_interface import (
    DeviceInterface,
    device_interfaces,
    register_interface_for_device,
)
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable


# ---------------------------------------------------------------------------
# 1. Fake DeviceInterface — directly inherits DeviceInterface.
#    Its current_stream raises at runtime, proving Dynamo always intercepts
#    it via the handler table during tracing.
# ---------------------------------------------------------------------------
class FakeDeviceInterface(DeviceInterface):
    @staticmethod
    def current_stream():
        raise RuntimeError(
            "FakeDeviceInterface.current_stream should never be called at "
            "runtime — Dynamo intercepts it via the handler table."
        )

    @staticmethod
    def stream(device_index):
        raise RuntimeError(
            "FakeDeviceInterface.stream should not be called directly"
        )

    @staticmethod
    def synchronize():
        pass


_REQUIRES_ACCELERATOR_REASON = (
    "Requires an accelerator (CUDA/XPU/NPU) to populate cur_stream_stack in "
    "SymbolicStreamState. Without it, handle_current_stream graph-breaks "
    "through its except clause."
)


@unittest.skipUnless(torch.accelerator.is_available(), _REQUIRES_ACCELERATOR_REASON)
class TestPrivateUse1StreamHandler(unittest.TestCase):
    """
    Regression test suite for https://github.com/pytorch/pytorch/pull/192970

    Verifies that a PrivateUse1 backend registered via
    ``register_interface_for_device()`` gets its ``current_stream`` mapped to
    ``handle_current_stream`` in Dynamo's handler table, fixing the
    ``streams::wait_stream(None)`` bug for out-of-tree backends.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # 1. Create and mount fake module
        cls.fake_device_module = types.ModuleType("fake_device")
        cls.fake_device_module.current_stream = FakeDeviceInterface.current_stream
        cls.fake_device_module.stream = FakeDeviceInterface.stream
        cls.fake_device_module.synchronize = FakeDeviceInterface.synchronize

        FakeDeviceInterface.current_stream.__module__ = "torch.fake_device"
        FakeDeviceInterface.current_stream.__qualname__ = "current_stream"

        torch.fake_device = cls.fake_device_module
        sys.modules["torch.fake_device"] = cls.fake_device_module

        # 2. Register interface
        register_interface_for_device("fake_device", FakeDeviceInterface)

        # 3. Register trace rule
        cls.torch_current_stream_fake = dict.fromkeys(
            ["torch.fake_device.current_stream"],
            TorchInGraphFunctionVariable,
        )
        trace_rules.torch_name_rule_map.append(cls.torch_current_stream_fake)
        trace_rules.clear_lru_cache()

        # 4. Clear Dynamo handler cache to force recomputation with fake_device
        if hasattr(TorchInGraphFunctionVariable._get_handlers, "cache_clear"):
            TorchInGraphFunctionVariable._get_handlers.cache_clear()

    @classmethod
    def tearDownClass(cls):
        # 1. Clear Dynamo handler cache to remove fake_device from future runs
        if hasattr(TorchInGraphFunctionVariable._get_handlers, "cache_clear"):
            TorchInGraphFunctionVariable._get_handlers.cache_clear()

        # 2. Remove trace rule
        if cls.torch_current_stream_fake in trace_rules.torch_name_rule_map:
            trace_rules.torch_name_rule_map.remove(cls.torch_current_stream_fake)
        trace_rules.clear_lru_cache()

        # 3. Remove interface registration (directly manipulate internal dict)
        if "fake_device" in device_interfaces:
            del device_interfaces["fake_device"]

        # 4. Clean up torch and sys.modules
        if hasattr(torch, "fake_device"):
            del torch.fake_device
        if "torch.fake_device" in sys.modules:
            del sys.modules["torch.fake_device"]

        super().tearDownClass()

    def test_wait_stream_compiles_in_graph(self):
        """Compile s.wait_stream(...) with fullgraph=True and assert 1 frame.

        This is the actual regression test: without the PR fix,
        torch.fake_device.current_stream is absent from the handler table,
        yielding a StreamVariable without user_object_index.  This causes
        ``streams::wait_stream(None)`` in the compiled graph, which raises
        a RuntimeError ("Expected int, got NoneType") at runtime.

        With the PR fix (or a manual fallback for unpatched torch), the
        handler resolves correctly, wait_stream receives a valid stream
        index, and compilation succeeds in a single frame with no graph
        break.
        """

        def fn(x):
            s = torch.fake_device.current_stream()
            s.wait_stream(torch.fake_device.current_stream())
            return x + 1

        cnt = CompileCounterWithBackend("inductor")
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)

        compiled_fn(torch.tensor(1.0))

        self.assertEqual(
            cnt.frame_count,
            1,
            "wait_stream should compile in-graph (1 frame). "
            "A graph break indicates the stream handler was not found in "
            "_get_handlers(), which is the regression this PR fixes.",
        )

    def test_base_class_stub_filtered(self):
        """Verify DeviceInterface.current_stream stub is NOT in handlers.

        Addresses review comment #3: the abstract base class defines
        ``current_stream`` (body raises ``NotImplementedError``), and
        backends like CpuInterface, MpsInterface, TpuInterface that do
        NOT override it inherit this stub.  The PR guards against
        registering the stub by checking
        ``_cs is not DeviceInterface.current_stream`` before insertion.
        """
        handlers = TorchInGraphFunctionVariable._get_handlers()
        base_cs = DeviceInterface.current_stream

        self.assertNotIn(
            base_cs,
            handlers,
            "Base class DeviceInterface.current_stream (the abstract stub "
            "that raises NotImplementedError) must be filtered out. "
            "It should never appear in the handler table.",
        )

    def test_fake_interface_current_stream_in_handlers(self):
        """Verify FakeDeviceInterface.current_stream appears in the handler table.

        On a PR-patched torch this is automatic: the iteration loop over
        ``get_registered_device_interfaces()`` in ``_get_handlers()``
        discovers the FakeDeviceInterface registered at module scope and
        adds its ``current_stream`` to the handler table.

        On an unpatched torch this assertion will fail — and that failure
        correctly documents that the PR fix is required.
        """
        handlers = TorchInGraphFunctionVariable._get_handlers()
        fake_cs = FakeDeviceInterface.current_stream

        self.assertIn(
            fake_cs,
            handlers,
            "FakeDeviceInterface.current_stream must be in the handler "
            "table.  With the PR this is automatic via the "
            "get_registered_device_interfaces() iteration loop.",
        )


if __name__ == "__main__":
    unittest.main()
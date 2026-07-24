"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.

Because the artifact is self-contained, re-executable Python, ``path`` is meant to
be committed and hand-edited: an engineer or agent can "hill-climb" the generated
kernel in place. This is ejectable compilation -- the emitted source is the source
of truth and is always exec'd, so hand edits always take effect. There is no
acceleration cache and no ``precompile.load`` round-trip: the source is exec'd as
written, so keeping the edited source correct is the caller's responsibility.
"""

import ast
import contextlib
import copy
import errno
import functools
import inspect
import logging
import operator
import os
import re
import secrets
import threading
from collections.abc import Callable, Sequence
from typing import Any, cast, NamedTuple, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Written as the artifact's first line so a later load can detect it was produced
# by a different torch (see _warn_on_version_skew). It is a comment, so it does not
# affect exec; a hand-edit that drops it just disables the skew warning, so
# hill-climbing an artifact never triggers a spurious version warning.
# All five stamps must stay in the artifact's LEADING comment block: the reader stops
# at the first line that is not a comment, so inserting code above them turns every
# check off -- each checked stamp warns per call while it is missing, and the version
# warning is the only one that goes quiet.
_VERSION_TAG = "# torch.compiler.export_python torch-version: "
# The train()/eval() flags of every nn.Module argument at capture. Python control flow
# on ``self.training`` is specialized into the graph with no runtime guard, so this
# stamp is what catches an artifact captured in one mode being run in the other. Like
# the version stamp it is exec-inert, and dropping it in a hand-edit just turns the
# check off (see _check_module_training).
_MODULE_TRAINING_TAG = "# torch.compiler.export_python module-training: "
# Which input tensors overlapped in memory at capture, which of them were literally the
# same object, and the ambient autocast state. make_fx bakes all three into the graph
# with no runtime guard: aliased inputs change what a mutation means, one object passed
# twice is deduped into a single graph slot, and autocast changes the dtypes the kernels
# were specialized for. Exec-inert like the other stamps, and dropping one turns only
# its own check off. What no stamp guards is a change in HOW two aliased inputs overlap
# -- capture's relative offsets stay baked in, exactly as they do under torch.compile.
_INPUT_OVERLAP_TAG = "# torch.compiler.export_python input-overlap: "
_INPUT_DUPLICATE_TAG = "# torch.compiler.export_python input-duplicates: "
_AUTOCAST_TAG = "# torch.compiler.export_python autocast: "
# Ambient process state the generated code bakes that no other stamp covers: the default
# dtype and device a factory op with no explicit argument resolves against, the matmul
# precision a GEMM template bakes as a constexpr, and whether deterministic algorithms
# were on when inductor chose between a deterministic and an atomic lowering.
_GLOBAL_STATE_TAG = "# torch.compiler.export_python global-state: "

# os.link failures that mean the filesystem cannot do hard links at all, as opposed to
# a real I/O problem (a full disk, a bad permission) that must not be swallowed.
_NO_HARDLINK_ERRNOS = frozenset(
    getattr(errno, name)
    for name in ("EPERM", "EOPNOTSUPP", "ENOTSUP", "EXDEV", "EMLINK", "ENOSYS")
    if hasattr(errno, name)
)


def _atomic_publish(path: str, data: bytes) -> bool:
    # Publish a fully-written file, never a partial one, and report whether this call
    # is the writer that published it. A hard link is the no-replace publish: exactly
    # one concurrent writer wins and every loser loads that winner rather than exec'ing
    # its own divergent source. Only errnos that mean "this filesystem has no hard
    # links" fall back to replace (last-writer-wins, still never partial); a full disk
    # or a permissions problem must surface rather than silently weaken the guarantee.
    dir_name = os.path.dirname(path) or "."
    base = os.path.basename(path)
    tmp = os.path.join(
        dir_name,
        f".{base}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp",
    )
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        except OSError as e:
            if e.errno not in _NO_HARDLINK_ERRNOS:
                raise
            log.warning(
                "torch.compiler.export_python: %s has no hard links (%s), so %s is "
                "published last-writer-wins; concurrent first writers may each run "
                "their own generated source.",
                dir_name,
                e.strerror,
                path,
            )
            os.replace(tmp, path)
        return True
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


# The buffer-donation contract, stated once here because out= is the one part of
# export_python that needs anything of the artifact beyond "it is runnable python".
# The decorated function opts into this calling convention by declaring a keyword-only
# ``out=None`` parameter; the artifact must then satisfy the following contract:
#
#   An artifact supports out= iff it takes every buffer it allocates from a callable
#   bound at module level in its own namespace under a name starting with
#   _ALLOCATOR_PREFIX. Rebinding those names is then enough to hand the generated code
#   memory instead of letting it allocate (see _BufferDonationPool).
#
# Nothing here is keyed on the backend: an artifact that meets the contract donates,
# one that does not is rejected at the first out= call. The inductor backend meets it
# by binding one allocator per device type (empty_strided_cuda, empty_strided_cpu,
# empty_strided_cpu_pinned, ...) regardless of which the graph uses; the eager backend
# emits a bare graph call and binds none. Allocators are matched by prefix rather than
# by an explicit list because the pool replays allocations BY ORDER: missing one
# device's allocator would desync the whole plan rather than fail. Both halves of that
# are pinned by tests -- test_allocator_prefix_matches_codegen_call_sites checks every
# allocator inductor's codegen can emit, and test_pool_intercepts_every_artifact_allocator
# checks every one a real artifact binds.
_ALLOCATOR_PREFIX = "empty_strided"
_DONATABLE_ALLOCATORS = {
    "empty_strided",
    "empty_strided_cpu",
    "empty_strided_cpu_pinned",
    "empty_strided_cuda",
    "empty_strided_mtia",
    "empty_strided_xpu",
}


def _precompile_error(msg: str) -> Exception:
    from torch._precompile import PrecompileError

    return PrecompileError(msg)


def _module_training_state(
    args: Sequence[Any],
) -> list[tuple[int, list[tuple[str, bool]]]]:
    return [
        (pos, [(name, module.training) for name, module in arg.named_modules()])
        for pos, arg in enumerate(args)
        if isinstance(arg, torch.nn.Module)
    ]


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    # The [start, end) byte range the tensor can touch. Exact for a dense tensor and a
    # bounding range for a strided one, which errs toward reporting overlap.
    start = t.data_ptr()
    extent = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return start, start + (extent + 1) * t.element_size()


def _dense_leaves(t: torch.Tensor) -> list[torch.Tensor] | None:
    """The real-memory tensors inside t, or None if its bytes cannot be located.

    A wrapper subclass (DTensor, TwoTensor, FunctionalTensor) reports data_ptr() == 0
    with a real device and is_meta False, so comparing its byte span against a plain
    tensor's would report every pair as disjoint. Decompose it instead.
    """
    if is_traceable_wrapper_subclass(t):
        try:
            attrs, _ = t.__tensor_flatten__()
        except Exception:
            return None
        leaves: list[torch.Tensor] = []
        saw_tensor = False
        for attr in attrs:
            # __tensor_flatten__ names the attributes that must be transformed, and
            # not all of them are tensors -- DTensor's list includes its DeviceMesh.
            component = getattr(t, attr, None)
            if not isinstance(component, torch.Tensor):
                continue
            saw_tensor = True
            inner = _dense_leaves(component)
            if inner is None:
                return None
            leaves.extend(inner)
        # An empty list here means every component owns no bytes, which is a real
        # answer. Only a subclass we could not see INTO is unresolvable -- returning
        # `leaves` unconditionally would make such a wrapper disjoint from everything
        # and let a donor be written over a live input.
        return leaves if saw_tensor else None
    if t.numel() == 0 or t.is_meta:
        # Owns no bytes, which is not the same as "bytes we cannot find". Both report
        # data_ptr 0, so without this one such component (an absent bias, an empty KV
        # cache, an uneven shard, a meta-initialized slot) would poison its whole
        # wrapper into "assume it aliases everything" and refuse every donation against
        # it. This is about a leaf that genuinely IS meta; a wrapper merely presenting
        # meta over live payloads is still distrusted, in _reports_no_bytes.
        return []
    try:
        if t.data_ptr() == 0:
            return None
    except RuntimeError:
        return None
    return [t]


def _reports_no_bytes(t: torch.Tensor) -> bool:
    """Whether t can be ruled out from its own report, without locating its bytes.

    Only for tensors that are what they say they are. A wrapper subclass's numel and
    is_meta describe what it PRESENTS: torch.load with
    map_location={torch.device("cpu"): "meta"} -- keyed by a device OBJECT, which remaps
    the wrapper without remapping its storages -- builds one reporting meta over live
    cpu payloads, and taking that at face value says it aliases nothing, including its
    own payload. The string form {"cpu": "meta"} does the reverse and is not the case
    this guards.
    """
    return not is_traceable_wrapper_subclass(t) and (t.numel() == 0 or t.is_meta)


def _shares_memory(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors can touch the same bytes.

    NOT torch._C._overlaps, which is IValue::overlaps -- storage IDENTITY, ignoring
    offsets. That reports byte-disjoint slices of one buffer as overlapping, which
    rejects the arena / fused-QKV / KV-cache shape this API exists to serve.

    Compares addresses, not storage objects: the same bytes can be reached through
    different UntypedStorages (from_numpy on overlapping slices, frombuffer, DLPack,
    __cuda_array_interface__ onto a live arena), and a storage-identity gate reports
    those as disjoint -- a donor would then be written over a live input. data_ptr is a
    process-global address and differing devices are rejected at the LEAF below, so
    distinct allocations cannot collide. One allocation visible under two device types
    still can: mapped pinned host memory has the same address as its CUDA view, and
    this reports the pair disjoint. torch._C._overlaps answers that case identically.
    Conservative for anything whose extent cannot be computed: a bounding range for
    strided tensors, and for sparse or otherwise unlocatable tensors "aliases anything
    of the same device type" -- NOT storage identity, which is the predicate this
    function exists to avoid.
    """
    if _reports_no_bytes(a) or _reports_no_bytes(b):
        # No real memory, and every meta tensor reports data_ptr 0, which would
        # otherwise make every pair look coincident.
        return False
    a_leaves, b_leaves = _dense_leaves(a), _dense_leaves(b)
    if a_leaves is None or b_leaves is None:
        # An addressless or undecomposable tensor (a wrapper subclass whose components
        # we cannot reach, sparse, nested). Its bytes are unknown, so the only safe
        # answer is "assume it aliases" -- NOT storage identity, which is the predicate
        # that made byte-disjoint arena slices look aliased in the first place. Device
        # type is all that can still rule a pair out. Note this reads the OUTER device
        # of both operands, including one that did resolve, because there is no leaf on
        # the unresolved side to pair its leaves against; a wrapper misreporting its
        # type is therefore still taken at its word here.
        return a.device.type == b.device.type
    if not (
        len(a_leaves) == 1
        and a_leaves[0] is a
        and len(b_leaves) == 1
        and b_leaves[0] is b
    ):
        return any(_shares_memory(x, y) for x in a_leaves for y in b_leaves)
    if a.device != b.device:
        # Compared here, on the LEAF, and never on the wrapper: a subclass can report a
        # device that differs from the one holding its bytes in index (torch.load with
        # map_location="cuda" over a "cuda:0" payload) or in TYPE (map_location=
        # {"cuda:0": "cpu"}). Gating above reported such a tensor as sharing memory with
        # nothing -- including with its own payload.
        return False
    try:
        (a_start, a_end), (b_start, b_end) = _byte_span(a), _byte_span(b)
    except RuntimeError:
        return True
    return a_start < b_end and b_start < a_end


def _input_tensors(args: Sequence[Any]) -> list[torch.Tensor]:
    """Every tensor an artifact call can reach, in a stable order.

    Module params and buffers are included, and come after the user tensors so their
    presence does not shift the indices a previously written stamp recorded. They have
    to be here: AOTAutograd dedups a user tensor that aliases a module buffer into one
    graph slot, so an artifact captured with that alias computes -- and mutates -- the
    wrong thing when the runtime call passes independent tensors, and the reverse.
    """
    user = [
        leaf
        for arg in args
        if not isinstance(arg, torch.nn.Module)
        for leaf in pytree.tree_leaves(arg)
        if isinstance(leaf, torch.Tensor)
    ]
    module: list[torch.Tensor] = []
    seen: set[int] = set()
    for arg in args:
        if not isinstance(arg, torch.nn.Module):
            continue
        for tensor in [*arg.parameters(), *arg.buffers()]:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                module.append(tensor)
    return [*user, *module]


def _span_atoms(
    t: torch.Tensor,
) -> list[tuple[torch.device, int, int]] | None:
    """t's byte ranges as (leaf device, start, end), or None if they cannot be located.

    Keyed on the LEAF device only, matching _shares_memory: a wrapper subclass can
    report a device that differs from the one holding its bytes, in index (torch.load
    with map_location="cuda" over a "cuda:0" payload) or in type (map_location=
    {"cuda:0": "cpu"}), so the wrapper's own report decides nothing. An empty list means
    t owns no addressable bytes, so it overlaps nothing.
    """
    if _reports_no_bytes(t):
        return []
    leaves = _dense_leaves(t)
    if leaves is None:
        return None
    atoms = []
    for leaf in leaves:
        if leaf.numel() == 0 or leaf.is_meta:
            continue
        try:
            start, end = _byte_span(leaf)
        except RuntimeError:
            return None
        atoms.append((leaf.device, start, end))
    return atoms


def _input_overlaps(
    args: Sequence[Any], tensors: list[torch.Tensor] | None = None
) -> list[list[int]]:
    """Which pairs of input tensors share memory, as sorted [i, j] index pairs.

    Runs on every artifact call, so it is a sweep and not the P-choose-2 pairwise scan:
    a 32-block model has hundreds of parameters, and the quadratic form cost multiples
    of the forward it guards. Lists (not tuples) so the stamp round-trips through
    ast.literal_eval to something that compares equal.
    """
    if tensors is None:
        tensors = _input_tensors(args)
    by_device: dict[torch.device, list[tuple[int, int, int]]] = {}
    unresolved: list[int] = []
    for i, tensor in enumerate(tensors):
        atoms = _span_atoms(tensor)
        if atoms is None:
            unresolved.append(i)
            continue
        for leaf, start, end in atoms:
            by_device.setdefault(leaf, []).append((start, end, i))
    pairs: set[tuple[int, int]] = set()
    for spans in by_device.values():
        spans.sort()
        # Spans that started earlier and have not ended: exactly the ones this span can
        # overlap, since the list is sorted by start.
        open_spans: list[tuple[int, int]] = []
        for start, end, i in spans:
            open_spans = [(e, j) for e, j in open_spans if e > start]
            for _, j in open_spans:
                if j != i:
                    pairs.add((min(i, j), max(i, j)))
            open_spans.append((end, i))
    for i in unresolved:
        for j, other in enumerate(tensors):
            if j != i and _shares_memory(tensors[i], other):
                pairs.add((min(i, j), max(i, j)))
    return [[i, j] for i, j in sorted(pairs)]


def _input_duplicates(
    args: Sequence[Any], tensors: list[torch.Tensor] | None = None
) -> list[list[int]]:
    """Which input positions hold the SAME tensor object, as [first, repeat] pairs.

    Not implied by _input_overlaps: AOTAutograd dedups arguments that are one object
    into a single graph slot, and byte overlap cannot tell that apart from two views
    that merely intersect. Both report the same pair set, so without this an artifact
    captured from overlapping views silently computes the wrong thing when handed one
    tensor twice. torch.compile guards it and recompiles ("Duplicate tensors found").
    """
    if tensors is None:
        tensors = _input_tensors(args)
    first: dict[int, int] = {}
    pairs = []
    for i, tensor in enumerate(tensors):
        seen_at = first.setdefault(id(tensor), i)
        if seen_at != i:
            pairs.append([seen_at, i])
    return pairs


def _code_devices(code: str) -> set[str]:
    """The device types the emitted artifact names, from its own source.

    The stamp and the per-call check have to agree, so this is computed from the source
    once at capture and once at load rather than from anything ambient.
    """
    found = set(re.findall(r"empty_strided_(\w+)\(", code))
    found.update(re.findall(r"device\(type=[\'\"](\w+)[\'\"]", code))
    found.update(re.findall(r"\.to\([\'\"](\w+)[\'\"]", code))
    devices = set()
    for name in found:
        # The allocator names are not all device types -- empty_strided_cpu_pinned is a
        # cpu allocator -- and a hand-edited artifact can contain anything at all.
        try:
            devices.add(torch.device(name).type)
        except (RuntimeError, ValueError):
            continue
    return devices


def _autocast_state(
    args: Sequence[Any],
    tensors: list[torch.Tensor] | None = None,
    code_devices: set[str] | None = None,
) -> list[list[Any]]:
    """Ambient autocast for every device type this artifact can compute on.

    The input devices alone are not enough: a graph whose inputs are all on CPU can still
    run its matmuls on an accelerator, and keying only on inputs recorded [] for it in
    both processes, so the check passed while the kernels had been built for autocast
    dtypes. Widened with the device types the emitted code names. Still not every device
    type in the process: a CPU helper first called inside a `with torch.autocast("cuda")`
    training region must not be locked to that region, and its graph never names cuda.
    """
    if tensors is None:
        tensors = _input_tensors(args)
    devices = {t.device.type for t in tensors}
    if code_devices is not None:
        devices.update(code_devices)
    return [
        [device, str(torch.get_autocast_dtype(device))]
        for device in sorted(devices)
        if torch.amp.is_autocast_available(device) and torch.is_autocast_enabled(device)
    ]


def _global_state() -> list[list[Any]]:
    """Ambient globals the emitted code resolves against, as sorted [key, value] pairs.

    Recorded as strings so the stamp round-trips through ast.literal_eval. These are all
    read at CAPTURE and baked: a factory op with no dtype= takes the default dtype then,
    a Triton GEMM template bakes ALLOW_TF32 as a tl.constexpr, and inductor picks a
    deterministic or an atomic-add lowering from the determinism flag. The artifact never
    re-consults any of them, so a process that changes one and replays gets capture's
    answer with no error.
    """
    return [
        ["default_dtype", str(torch.get_default_dtype())],
        ["default_device", str(torch.get_default_device())],
        ["deterministic", str(torch.are_deterministic_algorithms_enabled())],
        ["float32_matmul_precision", torch.get_float32_matmul_precision()],
    ]


def _check_donor_overlap(
    donated: Sequence[Any], inputs: Sequence[torch.Tensor]
) -> None:
    # Rechecked on every donating call, not just the recording one: the donors are
    # fixed by the recording but the inputs are whatever the caller passes, so an
    # input that aliases a donor is a fresh hazard each call.
    previous: list[torch.Tensor] = []
    for pos, donor in enumerate(donated):
        if not isinstance(donor, torch.Tensor):
            raise _precompile_error(
                f"torch.compiler.export_python: out[{pos}] must be a Tensor; got "
                f"{donor!r}."
            )
        if any(_shares_memory(donor, other) for other in [*inputs, *previous]):
            raise _precompile_error(
                f"torch.compiler.export_python: out[{pos}] overlaps an input or "
                "another donated output; donation requires disjoint storage on every "
                "call."
            )
        previous.append(donor)


def _check_served_buffer(
    served: torch.Tensor,
    request: tuple[Any, ...],
    slot: int,
    device: torch.device | None,
) -> None:
    """Verify a pooled or donated buffer against the allocation it is standing in for.

    device comes from the plan's record of where the artifact ALLOCATED this slot, not
    from the allocator name in the generated code -- empty_strided_cuda encodes only the
    device type, so deriving it there would still hand a cuda:1 buffer to a kernel built
    for cuda:0 -- and not from any tensor's current .device, since an opaque op holding
    a pooled buffer or a graph output can re-point it in place before we look.

    This covers the axes the generated code bakes in. It does NOT compare addresses, so
    a re-point that keeps size, stride, dtype and device but moves the memory is not
    detected -- at parity with eager out=, which likewise permits writing into a tensor
    that aliases an input, and with inductor's own assert_size_stride at point of use.
    """
    size = tuple(request[0]) if len(request) > 0 else None
    stride = tuple(request[1]) if len(request) > 1 else None
    dtype = request[2] if len(request) > 2 else None
    if (
        (size is not None and tuple(served.shape) != size)
        or (stride is not None and served.stride() != stride)
        or (isinstance(dtype, torch.dtype) and served.dtype != dtype)
        or (device is not None and served.device != device)
    ):
        raise _precompile_error(
            f"torch.compiler.export_python: the buffer for allocation {slot} no longer "
            f"matches the request the generated code makes for it (wanted size {size}, "
            f"stride {stride}, {dtype} on {device}; got size {tuple(served.shape)}, "
            f"stride {served.stride()}, {served.dtype} on {served.device}). The buffer "
            "was re-pointed after it was recorded -- a donor after it was validated, or "
            "the pool's own scratch by something holding it; the kernel writes through "
            "baked sizes "
            "and would run off the end."
        )


class _BufferDonationPool:
    """Serves the artifact's buffer allocations when the caller donates outputs.

    Takes the artifact as satisfying the donation contract above ``_ALLOCATOR_PREFIX``:
    every buffer comes from a module-level allocator in its own namespace, so rebinding
    those names is enough to hand it memory rather than let it allocate. The first
    donating call records the ordered requests and which of them come back as graph
    outputs; later donating calls replay that recording, serving scratch buffers from
    the pool and outputs from the caller, so a steady-state donating call allocates
    nothing.

    Calls that do not donate go straight through to the real allocator, so adding an
    ``out=`` call site never changes a sequential plain call. Donating calls are bound
    to their recording stream because the pooled scratch tensors carry no cross-stream
    synchronization.
    """

    def __init__(self, ns: dict[str, Any]) -> None:
        # The plan replays the RECORDED buffers, whose sizes were baked at record time.
        # A graph with a dynamic input dim sizes its scratch from the runtime symbol, so
        # replaying at a larger dim writes past the recorded allocation -- silently wrong
        # numerics, and the count check cannot see it because only the sizes differ.
        if any(
            dim is None
            for shape in ns.get("USER_INPUT_SHAPES") or ()
            if shape is not None
            for dim in shape
        ):
            raise _precompile_error(
                "torch.compiler.export_python: out= cannot be used with an artifact "
                "that has dynamic input dimensions (mark_unbacked); its scratch buffers "
                "are sized per call, so a recorded plan would serve undersized memory. "
                "Drop out= for this artifact."
            )
        allocators = [
            n for n, v in ns.items() if n.startswith(_ALLOCATOR_PREFIX) and callable(v)
        ]
        if not allocators:
            raise _precompile_error(
                "torch.compiler.export_python: out= needs the artifact to allocate its "
                f"buffers through {_ALLOCATOR_PREFIX}* callables bound in its own "
                "namespace, and this one binds none (backend='eager' emits a bare graph "
                "call). Use backend='inductor', or drop out=."
            )
        self.recorded = False
        # (buffer, -1) for a pooled scratch slot, (None, position) for one the caller
        # donates. Replayed in allocation order, which the specialized graph fixes.
        self._plan: list[tuple[torch.Tensor | None, int]] = []
        # One device per plan slot, read when the artifact ALLOCATED that buffer. A
        # buffer's own later .device cannot serve as its own expectation: inductor hands
        # pooled scratch and graph outputs to opaque ops, and .data= there re-points the
        # very objects the plan holds, so a reading taken afterwards records the damage.
        self._plan_devices: list[torch.device] = []
        self._slots: list[tuple[torch.Tensor, str, torch.device]] = []
        self._recording = False
        self._donated: Sequence[torch.Tensor] | None = None
        self._streams: dict[torch.device, Any] = {}
        self._i = 0
        # The thread whose donating call currently owns the plan. Allocations from any
        # other thread go to the real allocator: the pool is a per-call recording, so
        # serving a bystander from it would hand one thread another's out= tensor, and
        # a bystander allocating during the recording call would be written into the
        # plan and then scribbled on by every later donating call.
        # (pid, tid) rather than a bare thread id so a forked child never inherits a
        # live-looking claim from a thread it did not get.
        # ONE lock, held for the whole of a donating call. Everything else about who
        # owns the plan is derived from it. Earlier revisions hand-rolled a claim
        # protocol (owner tuple, pid staleness, a context manager, a pending-operand
        # slot) and each addition grew its own failure mode: a rejected caller could
        # tear down the in-flight one, two callers could cross operands, and a raise
        # inside the traced call could leave the plan claimed forever. A lock has none
        # of those states. It is not proof against an async exception: the interpreter
        # can deliver one between the acquire and the try that owns the release, which
        # strands the lock (documented on export_python). It is fail-closed -- a
        # stranded pool refuses later donating calls rather than serving wrong memory.
        self._busy = threading.Lock()
        # Set by the lock holder, read by _alloc. (pid, tid) because a forked child
        # deterministically reuses its parent's thread idents, so a bare tid would make
        # an unrelated child thread look like the owner.
        self._donating_thread: tuple[int, int] | None = None
        # True only while the donating call's OWN graph is running. A re-entrant call on
        # the owning thread (an opaque op calling back into the artifact) is otherwise
        # indistinguishable from the graph's own requests and would be handed the
        # donor's buffers. A flag rather than a counter because nested() only ever needs
        # to turn serving OFF -- a counter that got out of step could turn it on.
        self._serving = False
        self._expected: list[_ExpectedDonor] = []
        # Published LAST: once these are in the artifact's namespace another thread's
        # kernel can call _alloc, so every field it reads must already exist. Each
        # wrapper keeps its OWN real function -- the preamble binds one allocator per
        # device type regardless of which the graph uses, and falling back through the
        # wrong one would hand a CPU kernel device memory.
        for name in allocators:
            ns[name] = functools.partial(self._alloc, name, ns[name])

    def serving_here(self) -> bool:
        """Whether a donating call on THIS thread currently owns the plan."""
        return self._donating_thread == (os.getpid(), threading.get_ident())

    @contextlib.contextmanager
    def _in_flight(self, what: str):
        # Non-blocking: a second donating call, from another thread or re-entrant on
        # this one, is rejected rather than queued -- the pooled scratch is a single
        # recording and cannot be shared. Acquire/release is atomic, so a rejection
        # leaves the in-flight call untouched and a raise cannot strand the plan.
        if not self._busy.acquire(blocking=False):
            raise _precompile_error(
                "torch.compiler.export_python: a donating out= call is already in "
                f"flight for this artifact, so {what} cannot share the buffer plan. "
                "Donating calls are neither reentrant nor thread-safe."
            )
        self._donating_thread = (os.getpid(), threading.get_ident())
        self._serving = False
        try:
            yield
        finally:
            self._donating_thread = None
            self._serving = False
            self._donated = None
            self._recording = False
            self._slots = []
            self._busy.release()

    def _alloc(
        self,
        name: str,
        real: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not (self.serving_here() and self._serving):
            return real(*args, **kwargs)
        if self._recording:
            buf = real(*args, **kwargs)
            # buf.device is read HERE. By plan-commit time the recording graph has run,
            # and an opaque op it contains can have re-pointed this very object, which
            # would bake the poisoned device in as the expectation.
            self._slots.append((buf, name, buf.device))
            return buf
        if self._donated is None:
            return real(*args, **kwargs)
        i = self._i
        self._i = i + 1
        try:
            buf, pos = self._plan[i]
        except IndexError:
            # The graph is shape-specialized, so the recorded request sequence is
            # fixed; anything past it is not something to serve from the plan.
            return real(*args, **kwargs)
        # A plan slot is either a donor index or a recorded buffer, never None.
        served = cast(
            "torch.Tensor",
            self._donated[pos] if pos >= 0 else buf,  # type: ignore[index]
        )
        # Check against the request we were just handed, not against what the donor
        # looked like at validation time. Anything can have re-pointed it since (an
        # opaque op calling .data=/set_/resize_), and the kernel writes through the
        # baked sizes in this very request.
        # Every slot, donated or pooled, is checked against the device recorded when
        # the artifact allocated it -- the only reading taken before anything in the
        # graph could have re-pointed the object.
        _check_served_buffer(served, args, i, self._plan_devices[i])
        return served

    def record(
        self, call: Callable[[list[Any]], Any], args: list[Any], out: Sequence[Any]
    ) -> list[Any]:
        """Run one recording call, then fix the plan and validate the donated tensors."""
        with self._in_flight("the recording call"):
            # Snapshot here too: __call__ already hands us a tuple, but the pool is
            # reachable directly and every read of a caller sequence must be the same
            # read the kernel is served from.
            return self._record_locked(call, args, tuple(out))

    def _record_locked(
        self, call: Callable[[list[Any]], Any], args: list[Any], out: Sequence[Any]
    ) -> list[Any]:
        # Read BEFORE the recording graph runs. Everything the graph can touch has to be
        # sampled up front: an op inside it can re-point a donor or an input in place,
        # and a device read afterwards records where the tensor was left, not where the
        # kernels were built to write.
        external_devices = [
            t.device for t in [*out, *args] if isinstance(t, torch.Tensor)
        ]
        self._slots = []
        self._recording = True
        self._serving = True
        try:
            outs = list(call(args))
        except BaseException:
            self._slots = []
            raise
        finally:
            self._serving = False
            self._recording = False
        slots = self._slots
        try:
            if len(out) != len(outs):
                raise _precompile_error(
                    f"torch.compiler.export_python: out= has {len(out)} tensors but the "
                    f"artifact returns {len(outs)}."
                )
            slot_of = {id(t): (i, name) for i, (t, name, _dev) in enumerate(slots)}
            out_slot: dict[int, int] = {}
            expected_donors: dict[int, _ExpectedDonor] = {}
            donors: list[torch.Tensor] = []
            inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
            for pos, produced in enumerate(outs):
                slot = slot_of.get(id(produced))
                if slot is None:
                    raise _precompile_error(
                        f"torch.compiler.export_python: output {pos} is not a buffer the "
                        "artifact allocates, so there is no allocation to redirect into "
                        "your tensor: it is produced by an extern or fallback kernel "
                        "(a matmul, a convolution, a custom op), or it is a view of a "
                        "buffer, or an input passed through. Drop out= for this artifact."
                    )
                i, allocator = slot
                if i in out_slot:
                    raise _precompile_error(
                        f"torch.compiler.export_python: output {pos} aliases output "
                        f"{out_slot[i]}, so distinct out= destinations cannot represent "
                        "the artifact's output aliasing. Drop out= for this artifact."
                    )
                donor = out[pos]
                # Allocation-time device, not the produced tensor's: see above.
                expected = _ExpectedDonor.of(produced)._replace(device=slots[i][2])
                _check_donor(donor, expected, pos)
                expected_donors[pos] = expected
                donors.append(donor)
                out_slot[i] = pos
            # Every slot, not just the ones that become outputs. Pooling a special
            # allocation holds a strong reference to it for the artifact's lifetime,
            # which for symmetric memory wedges the alloc_id and breaks later PLAIN
            # calls -- the one thing the pool promises never to change.
            for i, (_buf, allocator, _dev) in enumerate(slots):
                if allocator not in _DONATABLE_ALLOCATORS:
                    where = (
                        f"output {out_slot[i]}"
                        if i in out_slot
                        else f"scratch buffer {i}"
                    )
                    raise _precompile_error(
                        f"torch.compiler.export_python: {where} is allocated by "
                        f"{allocator}, whose memory provenance cannot be replaced by an "
                        "ordinary out= tensor. Drop out= for this artifact."
                    )
            _check_donor_overlap(donors, inputs)
            plan: list[tuple[torch.Tensor | None, int]] = [
                (None, out_slot[i]) if i in out_slot else (buf, -1)
                for i, (buf, _name, _dev) in enumerate(slots)
            ]
            streams: dict[torch.device, Any] = {}
            for device in [*(dev for _buf, _name, dev in slots), *external_devices]:
                # meta owns no memory and binds to no stream, so there is nothing here
                # for the guard to protect; every other module-less device falls through
                # to the refusal below instead of a bare driver error.
                if device.type in ("cpu", "meta") or device in streams:
                    continue
                try:
                    device_module = torch.get_device_module(device.type)
                except RuntimeError:
                    device_module = None
                current_stream = getattr(device_module, "current_stream", None)
                if current_stream is None:
                    raise _precompile_error(
                        "torch.compiler.export_python: out= cannot safely reuse "
                        f"scratch buffers on {device.type}, which exposes no current "
                        "stream API. Drop out= for this artifact."
                    )
                streams[device] = current_stream(device)
            # This one call allocated its own outputs, so the donated tensors have to be
            # filled by hand. Every later donating call writes into them directly. The
            # fill is pool plumbing rather than part of fn, so it must not build an
            # autograd graph -- but it still bumps the donor's version counter, which is
            # what lets a saved-for-backward tensor diagnose the mutation.
            with torch.no_grad():
                for donor, produced in zip(donors, outs):
                    donor.copy_(produced)
        except BaseException:
            self._slots = []
            raise
        # Commit the plan only after recording, validation, and the initial copies all
        # succeed, so a rejected first call can be retried cleanly.
        self._plan = plan
        self._plan_devices = [dev for _buf, _name, dev in slots]
        self._streams = streams
        self._slots = []
        self._expected = [expected_donors[pos] for pos in range(len(outs))]
        self.recorded = True
        return donors

    @contextlib.contextmanager
    def donate(self, donated: Sequence[torch.Tensor], args: Sequence[Any]):
        """Hold the plan for one replaying donating call.

        Operands arrive as parameters and are snapshotted before use: routing them
        through the pool would let two callers cross them, and holding the caller's
        live sequence would make the donor check a TOCTOU. Validation happens inside
        the lock, so a rejection unwinds through the same finally that a success does.
        """
        with self._in_flight("this donating call"):
            donated = tuple(donated)  # snapshot BEFORE validating, not after
            if len(donated) != len(self._expected):
                raise _precompile_error(
                    f"torch.compiler.export_python: out= has {len(donated)} tensors but "
                    f"the recorded plan donates {len(self._expected)}."
                )
            for pos, (donor, expected) in enumerate(zip(donated, self._expected)):
                _check_donor(donor, expected, pos)
            _check_donor_overlap(
                donated, [arg for arg in args if isinstance(arg, torch.Tensor)]
            )
            for device, recorded in self._streams.items():
                current = torch.get_device_module(device.type).current_stream(device)
                if current != recorded:
                    raise _precompile_error(
                        "torch.compiler.export_python: donated calls must run on the "
                        f"same accelerator stream used to record the buffer plan "
                        f"({recorded}); got {current}."
                    )
            self._i = 0
            self._donated = donated
            self._serving = True
            yield
            self.check_complete(self._i)

    def nested(self, call: Callable[[list[Any]], Any], args: list[Any]) -> Any:
        """Run a call that must NOT be served from the plan."""
        outer, self._serving = self._serving, False
        try:
            return call(args)
        finally:
            self._serving = outer

    def check_complete(self, requested: int) -> None:
        # The plan is replayed by allocation ORDER, so a call that asked for a
        # different number of buffers than was recorded served the wrong memory. That
        # cannot happen for a shape-specialized graph, but if it ever does it must be
        # loud on the first donating call rather than quietly wrong forever. Checked
        # only after a call that returned, so it never masks a real exception.
        if requested != len(self._plan):
            raise _precompile_error(
                f"torch.compiler.export_python: the donated call requested {requested} "
                f"buffers but {len(self._plan)} were recorded; the buffer donation plan "
                "is out of sync. Drop out= for this artifact."
            )


class _ExpectedDonor(NamedTuple):
    """What a generated kernel bakes in about an output it will be handed."""

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    pinned: bool

    @staticmethod
    def of(produced: torch.Tensor) -> "_ExpectedDonor":
        return _ExpectedDonor(
            produced.shape,
            produced.stride(),
            produced.dtype,
            produced.device,
            produced.is_pinned(),
        )


def _describe_donor(donor: Any) -> str:
    """The rejected donor, rendered without calling anything it may not support.

    _ExpectedDonor.of reads stride() and is_pinned(), which raise for sparse and nested
    layouts -- so formatting the refusal for one of those escaped as a raw ATen error
    instead of the message being built. The type name leads because a subclass whose
    metadata matches otherwise prints identically to what it is being refused against.
    """
    if not isinstance(donor, torch.Tensor):
        return repr(donor)
    name = type(donor).__name__
    if donor.is_nested:
        return f"a nested {name}"
    if donor.layout is not torch.strided:
        return f"a {name} with {donor.layout} layout"
    try:
        return f"{name} {_ExpectedDonor.of(donor)}"
    except RuntimeError:
        return name


def _check_donor(donor: Any, expected: _ExpectedDonor, pos: int) -> None:
    # Run in full on EVERY donating call. An earlier version skipped this when the
    # donor object was the same as last time, which is unsound: .data assignment,
    # set_() and resize_() all change a tensor's metadata (and storage) in place under
    # a stable id(), and id() is itself recycled after a free. Getting that wrong is
    # not a wrong number, it is a kernel writing through baked sizes into the wrong
    # allocation, so this is a handful of attribute reads the fast path has to pay.
    # Mirror eager's actual rule rather than the flag alone: torch.add(..., out=leaf)
    # is refused only with grad enabled, and an inference tensor is refused only outside
    # InferenceMode. This buys a plain leaf that requires grad, donated under no_grad --
    # NOT an nn.Parameter, which the exact-type test below refuses in every grad mode.
    if donor.__class__ is torch.Tensor:
        if donor.requires_grad and torch.is_grad_enabled():
            raise _precompile_error(
                f"torch.compiler.export_python: out[{pos}] requires grad and the kernel "
                "writes its memory directly, which eager's out= also refuses here. "
                "Donate under torch.no_grad(), or pass a donor that does not require grad."
            )
        if donor.is_inference() and not torch.is_inference_mode_enabled():
            raise _precompile_error(
                f"torch.compiler.export_python: out[{pos}] is an inference tensor being "
                "written outside InferenceMode, which ATen also refuses. Donate from "
                "inside the torch.inference_mode() block that allocated it."
            )
    if (
        type(donor) is not torch.Tensor
        or donor.layout is not torch.strided
        or donor.is_nested
        or donor.is_conj()
        or donor.is_neg()
        or donor.shape != expected.shape
        or donor.stride() != expected.stride
        or donor.dtype != expected.dtype
        or donor.device != expected.device
        or (expected.pinned and not donor.is_pinned())
    ):
        raise _precompile_error(
            f"torch.compiler.export_python: out[{pos}] must be a plain Tensor matching "
            f"the artifact's output exactly ({expected}, strided layout, no "
            "conjugate/negative view bits); got " + _describe_donor(donor) + "."
        )
    # Triton specializes on pointers being divisible by 16, so a donor the caching
    # allocator did not hand out (a misaligned view, an externally-owned buffer) can
    # silently miscompute under a kernel compiled against that assumption.
    from torch._inductor.utils import GPU_ALIGN_BYTES, is_gpu

    data_ptr = donor.data_ptr()
    if is_gpu(donor.device.type) and data_ptr % GPU_ALIGN_BYTES:
        raise _precompile_error(
            f"torch.compiler.export_python: out[{pos}] must have a {GPU_ALIGN_BYTES}-byte "
            f"aligned data pointer for the generated kernel; got {data_ptr}."
        )


def _lean_entry(forward: Callable[..., Any]) -> Callable[..., Any] | None:
    """Bind the artifact's compiled ``call`` directly, skipping the driver's guards.

    The emitted ``forward`` spends most of its time re-verifying the precompile
    invariants on every call (pytree round-trip, per-input shape/dtype/device guards,
    module structure). Everything it guards is fixed at capture, so a caller that
    accepts the invariants can call the compiled ``call`` directly. Returns None when
    the driver is doing real marshalling rather than checking -- an ``nn.Module``
    argument to lift params out of, a gradient to scatter, or an input/output
    structure that is not a flat sequence of leaves -- in which case there is nothing
    safe to strip and the caller keeps ``forward``.
    """
    # forward was exec'd into the artifact's own module namespace, so its __globals__
    # IS that namespace: the composed ``call`` and the baked calling convention are
    # reachable through it without re-exec'ing or re-parsing the source.
    ns = forward.__globals__
    if ns.get("MODULE_POSITIONS") or ns.get("GRAD_PARAM_INDICES"):
        return None
    try:
        # ``call`` is AOTAutograd's composed entry point, NOT the raw inductor one:
        # it still reflects input mutation, unwraps subclasses and disables grad. Only
        # the precompile driver's guards are dropped, never AOTAutograd's semantics.
        call = ns["call"]
        in_spec_str, out_spec_str = ns["IN_SPEC"], ns["OUT_SPEC"]
    except KeyError:
        return None
    if in_spec_str is None or out_spec_str is None:
        return None
    in_spec = pytree.treespec_loads(in_spec_str)
    if in_spec.type is not tuple or not all(c.is_leaf() for c in in_spec.children()):
        return None
    out_spec = pytree.treespec_loads(out_spec_str)
    if out_spec.is_leaf():
        rebuild: Callable[[list[Any]], Any] = operator.itemgetter(0)
    elif out_spec.type in (tuple, list) and all(
        c.is_leaf() for c in out_spec.children()
    ):
        rebuild = out_spec.type
    else:
        return None
    num_args = in_spec.num_children
    set_grad, grad_enabled = torch._C._set_grad_enabled, torch.is_grad_enabled
    # Only the eager driver runs its call under no_grad (an eager ``call`` is a bare
    # graph call, so grad has to be disabled to match). The inductor driver runs under
    # the AMBIENT grad mode on purpose: AOTAutograd's output-alias epilogue is outside
    # the inner no_grad and is grad-mode-sensitive, so forcing grad off here silently
    # strips grad_fn from an output that aliases an input.
    disable_grad = ns.get("BACKEND") == "eager"
    # Built on the first out= call, under a lock: the pool's own lock lives inside the
    # object being built, so two threads racing here would each get a pool with its own
    # lock, stack a second allocator wrapper over the first permanently, and both run
    # a recording call at once.
    pool: _BufferDonationPool | None = None
    pool_lock = threading.Lock()

    def lean_forward(*args: Any, out: Sequence[torch.Tensor] | None = None) -> Any:
        nonlocal pool
        # Arity is checked because it costs ~30ns and the alternative is an unpack
        # ValueError raised from generated source. Nothing else is: shapes, dtypes,
        # devices and layouts are the caller's responsibility in this mode.
        if len(args) != num_args:
            raise _precompile_error(
                f"precompile: expected {num_args} positional args (the same as the "
                f"traced fn), got {len(args)}."
            )
        if out is not None and pool is None:
            with pool_lock:
                if pool is None:
                    pool = _BufferDonationPool(ns)
        grad = disable_grad and grad_enabled()
        if grad:
            set_grad(False)
        try:
            if out is None:
                if pool is None or not pool.serving_here():
                    return rebuild(call(list(args)))
                # A plain call nested inside this artifact's own donating call must not
                # be served from the plan, so it runs one level deeper. Only the owner's
                # nesting counts, since the flag is shared state.
                return rebuild(pool.nested(call, list(args)))
            if not pool.recorded:  # type: ignore[union-attr]
                return rebuild(pool.record(call, list(args), out))  # type: ignore[union-attr]
            with pool.donate(out, args):  # type: ignore[union-attr]
                torch.autograd.graph.increment_version(out)
                result = rebuild(call(list(args)))
            return result
        finally:
            if grad:
                set_grad(True)

    return lean_forward


class ExportedPythonArtifact:
    """Materializes and disk-caches a ``torch.compiler.precompile`` artifact.

    Materialization is lazy and happens on the first call: if ``path`` exists the
    emitted Python is read from disk, otherwise the wrapped ``fn`` is precompiled
    against the example inputs and the emitted source is written to disk. Either
    way the source is exec'd directly to build the runnable. The loaded callable is
    reused for all subsequent calls in the process; a later process re-reads
    whatever is on disk.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        path: str,
        backend: str,
        tracer: str,
        decompositions: dict | None,
        example_inputs: Sequence[object] | None,
        unsafe_reduce_overhead: bool = False,
    ) -> None:
        self._fn = fn
        self._signature = inspect.signature(fn)
        out_param = self._signature.parameters.get("out")
        self._declares_out = (
            out_param is not None
            and out_param.kind == inspect.Parameter.KEYWORD_ONLY
            and out_param.default is None
        )
        self._call_signature = (
            self._signature.replace(
                parameters=[
                    param
                    for name, param in self._signature.parameters.items()
                    if name != "out"
                ]
            )
            if self._declares_out
            else self._signature
        )
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._unsafe_reduce_overhead = unsafe_reduce_overhead
        self._module_training: list[tuple[int, list[tuple[str, bool]]]] | None = None
        self._input_overlaps: list[list[int]] | None = None
        self._input_duplicates: list[list[int]] | None = None
        self._global_state: list[list[str]] | None = None
        self._code_devices: set[str] = set()
        self._autocast: list[list[Any]] | None = None
        # Whether the lean entry point (the one that accepts out=) is what got bound;
        # unsafe_reduce_overhead can fall back to the checked forward at load time.
        self._lean_bound = False
        self._loaded: Callable[..., Any] | None = None
        # (pid, tid) currently inside _materialize. There is deliberately no
        # per-artifact lock: capture is already serialized process-wide, so a second
        # lock would add nothing but a second acquisition order to deadlock against.
        # This is the re-entrancy guard the (reentrant) capture lock cannot provide.
        # The pid makes it fork-safe without a registry -- a child never matches a
        # marker left by a thread it did not inherit.
        self._materializing: tuple[int, int] | None = None

    def _precompile_and_save(self, args: tuple[Any, ...]) -> tuple[str, bool]:
        example = self._example_inputs
        if example is None:
            # Capture runs fn once on the example inputs (real-mode make_fx), which
            # mutates them; deep-copy the live call args so capture side effects (in-
            # place input mutation, module buffer updates) do not leak onto the
            # caller before the artifact itself runs on the real args exactly once.
            try:
                example = copy.deepcopy(args)
            except Exception as e:
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    "torch.compiler.export_python could not deep-copy the "
                    "first-call arguments to capture without mutating them (e.g. a "
                    "non-leaf tensor or a weight_norm module). Pass explicit "
                    "example_inputs=... to precompile against dedicated inputs."
                ) from e
            if _input_overlaps(example) != _input_overlaps(args):
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    "torch.compiler.export_python: deep-copying the first-call "
                    "arguments did not preserve how their tensors share memory, so "
                    "capturing from the copy would bake in aliasing the real arguments "
                    "do not have. nn.Parameter.__deepcopy__ clones, so two Parameters "
                    "backed by one storage become independent in the copy. Pass "
                    "example_inputs=... built with the same sharing as the real "
                    "arguments to capture against those instead."
                )
        else:
            example = self._bind_positional(example, {}, "example_inputs=")
            self._check_supported_args(example)
        # precompile returns (python_code, cache); the cache is an acceleration
        # artifact that export_python does not use -- the emitted source is
        # self-contained and always exec'd -- so only the code is written to disk.
        code, _cache = torch.compiler.precompile(
            self._fn,
            *example,
            backend=self._backend,
            tracer=self._tracer,
            decompositions=self._decompositions,
        )
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # The stamps lead the artifact as exec-inert comments, each guarding one thing
        # make_fx specialized without a runtime guard. A hand-edit may drop any of them;
        # each just turns its own check off.
        example_tensors = _input_tensors(example)
        code = (
            f"{_VERSION_TAG}{torch.__version__!r}\n"
            f"{_MODULE_TRAINING_TAG}{_module_training_state(example)!r}\n"
            f"{_INPUT_OVERLAP_TAG}{_input_overlaps(example, example_tensors)!r}\n"
            f"{_INPUT_DUPLICATE_TAG}{_input_duplicates(example, example_tensors)!r}\n"
            f"{_AUTOCAST_TAG}"
            f"{_autocast_state(example, example_tensors, _code_devices(code))!r}\n"
            f"{_GLOBAL_STATE_TAG}{_global_state()!r}\n{code}"
        )
        if _atomic_publish(self._path, code.encode("utf-8")):
            return code, False
        # Lost the publish race; the winner's file is complete and already linked.
        winner = self._load_from_disk()
        if winner is None:
            raise _precompile_error(
                f"torch.compiler.export_python: another writer published {self._path} "
                "and it was deleted before this call could load it. Retry."
            )
        return winner, True

    def _load_from_disk(self) -> str | None:
        # None means "not there after all" -- the presence gate raced a peer deleting
        # the artifact to force a regenerate, which should fall through to capture
        # rather than surface a bare FileNotFoundError.
        try:
            with open(self._path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise _precompile_error(
                f"torch.compiler.export_python: could not read the artifact at "
                f"{self._path} ({e.strerror}). Check that the path names a readable "
                "file rather than a directory."
            ) from e

    @staticmethod
    def _read_raw_stamp(code: str, tag: str) -> str | None:
        for line in code.splitlines():
            if line.startswith(tag):
                return line[len(tag) :].strip() or None
            # An INDENTED comment is still a comment. The documented rule is "the
            # first non-comment line", and stopping early here silently turns every
            # later stamp's check off rather than reading it.
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                break
        return None

    @staticmethod
    def _read_stamp(code: str, tag: str) -> Any:
        for line in code.splitlines():
            if line.startswith(tag):
                try:
                    return ast.literal_eval(line[len(tag) :])
                except (ValueError, SyntaxError):
                    # A mangled stamp is a hand-edit like any other: turn the check off
                    # rather than raise a SyntaxError from a comment line that does not
                    # affect what the artifact runs.
                    return None
            # An INDENTED comment is still a comment. The documented rule is "the
            # first non-comment line", and stopping early here silently turns every
            # later stamp's check off rather than reading it.
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                break
        return None

    def _check_capture_environment(self, args: tuple[Any, ...]) -> None:
        # Two more things make_fx specialized with no runtime guard. Input ALIASING
        # decides what an in-place mutation means, so an artifact captured with two
        # arguments sharing memory computes the wrong thing (and mutates the wrong
        # thing) when they are distinct at runtime -- torch.compile guards this and
        # recompiles. Ambient AUTOCAST decides the dtypes the kernels were built for.
        tensors: list[torch.Tensor] | None = None

        def input_tensors() -> list[torch.Tensor]:
            # Walked once and shared by the three checks below. Lazily, so an artifact
            # whose stamps were all hand-edited away does not pay for a walk no check
            # will read.
            nonlocal tensors
            if tensors is None:
                tensors = _input_tensors(args)
            return tensors

        if self._input_overlaps is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "input-aliasing stamp, so a runtime call whose inputs share memory "
                "differently than capture is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual = _input_overlaps(args, input_tensors())
            if actual != self._input_overlaps:
                raise _precompile_error(
                    "torch.compiler.export_python: the runtime inputs do not share "
                    f"memory the way capture did (captured overlapping index pairs "
                    f"{self._input_overlaps}, got {actual}). Aliasing is baked into the "
                    "graph, so this call would compute against the wrong assumption."
                )
        if self._input_duplicates is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "input-duplicate stamp, so a runtime call that repeats a tensor object "
                "differently than capture is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual_duplicates = _input_duplicates(args, input_tensors())
            if actual_duplicates != self._input_duplicates:
                raise _precompile_error(
                    "torch.compiler.export_python: the runtime inputs repeat tensor "
                    "objects differently than capture did (captured duplicate index "
                    f"pairs {self._input_duplicates}, got {actual_duplicates}). "
                    "AOTAutograd folds arguments that are one object into a single "
                    "graph slot, so this call would compute against the wrong "
                    "assumption -- byte overlap alone cannot see the difference."
                )
        if self._global_state is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "global-state stamp, so calling it under a different default dtype, "
                "default device, matmul precision or determinism setting than capture "
                "is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual_state = dict(_global_state())
            for key, captured in self._global_state:
                live = actual_state.get(key)
                if live == captured:
                    continue
                # Determinism is one-sided: capture with it OFF and replay with it ON
                # means the artifact keeps a lowering the caller has asked not to run.
                # ON at capture and OFF at replay is conservative, so it is not an error.
                if key == "deterministic" and captured == "True":
                    continue
                raise _precompile_error(
                    f"torch.compiler.export_python: {key} is {live} but the artifact "
                    f"was captured with {key} {captured}. It is resolved when the code "
                    "is generated and baked in, so this call would silently get "
                    "capture's answer. Set it to the captured value, or delete "
                    f"{self._path} to recapture."
                )
        if self._autocast is None:
            log.warning(
                "torch.compiler.export_python: the artifact at %s carries no recorded "
                "autocast stamp, so calling it under a different autocast context than "
                "capture is unchecked. Delete %s to regenerate it.",
                self._path,
                self._path,
            )
        else:
            actual_autocast = _autocast_state(args, input_tensors(), self._code_devices)
            if actual_autocast != self._autocast:
                raise _precompile_error(
                    "torch.compiler.export_python: the runtime autocast state does not "
                    f"match capture (captured {self._autocast}, got {actual_autocast}). "
                    "Autocast dtypes are baked into the artifact; capture under the "
                    "same autocast context you call it in."
                )

    def _check_module_training(self, args: tuple[Any, ...]) -> None:
        # A lean-bound artifact has no MODULE_POSITIONS (that is one of the things
        # _lean_entry refuses to strip), so it can never be handed an nn.Module and
        # this walk would be pure per-call cost on the path that exists to avoid it.
        if self._lean_bound:
            return
        actual = _module_training_state(args)
        if not actual:
            return
        if self._module_training is None:
            # Missing stamp means a hand-edit dropped it, the same as the version
            # stamp: warn that the guard is off rather than refuse to run an artifact
            # whose source is by design the thing the caller is free to edit.
            log.warning(
                "torch.compiler.export_python: the artifact at %s takes nn.Module "
                "arguments but carries no recorded training state, so train()/eval() "
                "skew against capture is unchecked. Delete %s to regenerate the stamp.",
                self._path,
                self._path,
            )
            return
        if actual != self._module_training:
            raise _precompile_error(
                "torch.compiler.export_python: the runtime module training state does "
                f"not match capture (expected {self._module_training!r}, got {actual!r}). "
                "Restore train()/eval() state or regenerate the artifact."
            )

    def _warn_on_version_skew(self, code: str) -> None:
        # Warn (but still run) when the artifact carries a version stamp that does
        # not match the current torch, so a committed artifact gone stale across a
        # torch upgrade is visible rather than silently running old logic. A missing
        # stamp (dropped by a hand-edit) is silent, so hill-climbing never warns.
        produced = self._read_stamp(code, _VERSION_TAG)
        if produced is None:
            # Artifacts written before the stamp became a repr() carry a bare version
            # string, which literal_eval rejects. Falling back to the raw text keeps the
            # skew warning working for every artifact already committed -- silently
            # losing it is exactly the case the stamp exists for.
            produced = self._read_raw_stamp(code, _VERSION_TAG)
        # str(): TorchVersion.__eq__ PEP-440-parses its operand and re-raises anything
        # that is not InvalidVersion, so a stamp of 4300+ digits took the whole load path
        # down with a ValueError. Comparing text keeps a mangled stamp to a warning.
        if produced is None or produced == str(torch.__version__):
            return
        log.warning(
            "torch.compiler.export_python: the artifact at %s was produced by "
            "torch %s but the current torch is %s; running it as-is. Delete %s "
            "to regenerate against the current torch.",
            self._path,
            produced,
            torch.__version__,
            self._path,
        )

    def _load(self, code: str, *, from_disk: bool) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip). A clobbered hand-edit (dropped forward / syntax
        # error) and an environment or version mismatch (an import that fails under the
        # current torch) surface as distinct, actionable PrecompileErrors rather than
        # one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        try:
            if from_disk:
                log.warning(
                    "torch.compiler.export_python is about to EXEC the artifact at %s; "
                    "the file is trusted executable Python and may have been edited or "
                    "replaced since export. Only load paths whose contents you trust.",
                    self._path,
                )
            return _make_inlined_forward(code, warn=False, filename=self._path)
        except (SyntaxError, KeyError) as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} could "
                "not be run as precompile source; it is not a valid "
                "torch.compiler.precompile artifact (a hand-edit may have clobbered "
                "it, e.g. dropping forward()). Delete it to regenerate."
            ) from e
        except ImportError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} failed "
                "to import a dependency; it was likely produced by a different torch "
                f"version or environment. Delete {self._path} to regenerate against "
                "the current torch."
            ) from e
        except Exception as e:
            raise PrecompileError(
                "torch.compiler.export_python: an unexpected error occurred running "
                f"the artifact at {self._path}. Delete it to regenerate."
            ) from e

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        code = self._load_from_disk() if os.path.exists(self._path) else None
        from_disk = code is not None
        if code is None:
            code, from_disk = self._precompile_and_save(args)
        if from_disk:
            self._warn_on_version_skew(code)
        self._module_training = self._read_stamp(code, _MODULE_TRAINING_TAG)
        self._input_overlaps = self._read_stamp(code, _INPUT_OVERLAP_TAG)
        self._input_duplicates = self._read_stamp(code, _INPUT_DUPLICATE_TAG)
        self._autocast = self._read_stamp(code, _AUTOCAST_TAG)
        self._global_state = self._read_stamp(code, _GLOBAL_STATE_TAG)
        self._code_devices = _code_devices(code)
        forward = self._load(code, from_disk=from_disk)
        entry = forward
        if self._unsafe_reduce_overhead:
            # A fallback is a warning rather than an error: the artifact still runs, the
            # flag just bought nothing, and failing a working call over a missed
            # optimization would be worse than saying so.
            lean = _lean_entry(forward)
            if lean is None:
                log.warning(
                    "torch.compiler.export_python: unsafe_reduce_overhead=True had no "
                    "effect for %s; its calling convention needs the artifact's driver "
                    "(an nn.Module argument, a gradient to scatter, or a non-flat "
                    "input/output structure). Running the checked entry point.",
                    self._path,
                )
            else:
                self._lean_bound = True
                entry = lean
        self._example_inputs = None
        self._decompositions = None
        return entry

    def _materialize_once(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        # Materialization runs under the one process-wide capture lock. Capture runs
        # fn, which may call another decorated function, so any second lock taken
        # around this would give two threads two orders to acquire them in and deadlock
        # -- which is why the artifact holds no lock of its own.
        import torch._precompile as precompile_impl

        ident = (os.getpid(), threading.get_ident())
        if self._materializing == ident:
            raise _precompile_error(
                "torch.compiler.export_python: re-entrant call into "
                f"{getattr(self._fn, '__name__', 'fn')} while it is being precompiled. "
                "A decorated function cannot call itself: capture would have to run "
                "inside its own capture. Move the recursion into an undecorated helper."
            )
        with precompile_impl._CAPTURE_LOCK:
            if self._loaded is None:
                self._materializing = ident
                try:
                    self._loaded = self._materialize(args)
                finally:
                    self._materializing = None
            return self._loaded

    def _bind_positional(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        source: str = "the call arguments",
    ) -> tuple[Any, ...]:
        # The artifact's forward is positional (the precompile calling convention),
        # so map any keyword call args onto fn's positional parameters -- this lets
        # callers invoke the decorated fn naturally (e.g. rope(q=..., k=...)).
        # Anything that cannot be laid out positionally is rejected below.
        sig = self._call_signature
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                f"torch.compiler.export_python: could not bind {source} to "
                f"{getattr(self._fn, '__name__', 'fn')}'s signature: {e}"
            ) from e
        bound.apply_defaults()
        # bound.kwargs holds every argument bind() could not place positionally. That
        # is a keyword-only / **kwargs param (never positional), or a plain
        # positional-or-keyword param passed by keyword while an earlier one was left
        # to its default -- distinguish them so the error names the real cause.
        if bound.kwargs:
            params = sig.parameters
            kw_only = sorted(
                n
                for n in bound.kwargs
                if n in params and params[n].kind == inspect.Parameter.KEYWORD_ONLY
            )
            if kw_only:
                raise TypeError(
                    "torch.compiler.export_python does not support keyword-only "
                    f"parameters (got {kw_only}); the precompile calling convention "
                    "is positional."
                )
            # Names not declared as parameters were absorbed by a **kwargs param;
            # they are never positional, so name **kwargs as the cause rather than
            # misreporting them as a positional-or-keyword arg left to its default.
            var_kw = sorted(n for n in bound.kwargs if n not in params)
            if var_kw:
                raise TypeError(
                    "torch.compiler.export_python does not support **kwargs "
                    f"parameters (got {var_kw}); the precompile calling convention "
                    "is positional."
                )
            raise TypeError(
                "torch.compiler.export_python could not place keyword arguments "
                f"{sorted(bound.kwargs)} positionally because an earlier positional "
                "parameter was left to its default; pass those arguments positionally "
                "or provide example_inputs."
            )
        return bound.args

    def _check_supported_args(self, args: tuple[Any, ...]) -> None:
        params = list(self._call_signature.parameters)
        for pos, arg in enumerate(args):
            if isinstance(arg, torch.nn.Module):
                continue
            unsupported = [
                leaf
                for leaf in pytree.tree_leaves(arg)
                if not isinstance(leaf, torch.Tensor)
            ]
            if not unsupported:
                continue
            name = params[pos] if pos < len(params) else f"argument {pos}"
            # These two land often enough that the generic "close the constant over"
            # advice is actively wrong for them: a module must stay an argument, and an
            # optional parameter has no constant to close over in the first place.
            if any(isinstance(leaf, torch.nn.Module) for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python: nn.Module arguments must be passed "
                    f"directly, not nested inside a container (parameter {name!r}). "
                    "Pass the module itself as its own positional argument."
                )
            if all(leaf is None for leaf in unsupported):
                raise TypeError(
                    "torch.compiler.export_python does not support None arguments "
                    f"(parameter {name!r}); make_fx specializes the None branch without "
                    "a runtime guard. Split the function, or pass a tensor."
                )
            raise TypeError(
                "torch.compiler.export_python supports only Tensor pytrees and "
                "nn.Module positional arguments; Python scalar/config values are "
                "specialized by make_fx without runtime guards. Close constants "
                f"over in the function instead of passing parameter {name!r} "
                f"({unsupported[0]!r})."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # The keyword-only out=None declaration makes donation visible in fn's
        # signature, but export_python intercepts the value before binding because it
        # belongs to the artifact calling convention, not the captured graph.
        # Only intercept ``out`` when fn opted in; otherwise a genuine parameter that
        # happens to be named ``out`` could not be passed by keyword at all.
        out = kwargs.pop("out", None) if self._declares_out else None
        if "out" in kwargs and "out" not in self._call_signature.parameters:
            raise TypeError(
                "torch.compiler.export_python: out= requires the decorated function "
                "to declare a keyword-only `out=None` parameter."
            )
        if out is not None and not self._unsafe_reduce_overhead:
            raise TypeError(
                "torch.compiler.export_python: out= requires "
                "unsafe_reduce_overhead=True; the checked entry point allocates its "
                "own outputs."
            )
        if isinstance(out, torch.Tensor):
            # len() on a tensor is its first dimension, so without this the arity
            # check downstream reports a nonsense count.
            raise TypeError(
                "torch.compiler.export_python: out= must be a sequence of tensors, one "
                "per artifact output, even when there is only one; got a Tensor."
            )
        if out is not None:
            # ONE read of the caller's sequence, here, at the boundary. Everything
            # downstream -- validation, the allocator, the version bump -- uses this
            # tuple. Re-reading it later is how a donor gets validated and a different
            # one written: a sequence is free to return something new each time.
            out = tuple(out)
        args = self._bind_positional(args, kwargs)
        self._check_supported_args(args)
        loaded = self._loaded
        if loaded is None:
            loaded = self._materialize_once(args)
        if not self._lean_bound:
            # unsafe_reduce_overhead makes every one of these the caller's
            # responsibility, exactly as it does for shape, dtype, device and layout.
            self._check_capture_environment(args)
        self._check_module_training(args)
        if out is None:
            return loaded(*args)
        if not self._lean_bound:
            raise TypeError(
                "torch.compiler.export_python: out= is not available for this "
                "artifact; unsafe_reduce_overhead had no effect for its calling "
                "convention (see the warning logged on load)."
            )
        return loaded(*args, out=out)


def export_python(
    *,
    path: str,
    backend: str = "inductor",
    tracer: str = "make_fx",
    decompositions: dict | None = None,
    example_inputs: Sequence[object] | None = None,
    unsafe_reduce_overhead: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """See :func:`torch.compiler.export_python`."""

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        artifact = ExportedPythonArtifact(
            fn,
            path=path,
            backend=backend,
            tracer=tracer,
            decompositions=decompositions,
            example_inputs=example_inputs,
            unsafe_reduce_overhead=unsafe_reduce_overhead,
        )

        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return cast("_R", artifact(*args, **kwargs))

        return wrapped

    return decorator

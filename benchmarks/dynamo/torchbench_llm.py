"""Decomposed LLM benchmark harness for the dynamo benchmark suite.

``TextGenerationBenchmark`` in ``huggingface_llm_models.py`` only hands back a
model and a batch of inputs; the generation loop itself lives inside
HuggingFace's ``model.generate()``. That makes it impossible to attribute cost
to the two phases that matter for inference -- the compute-bound *prefill* and
the memory-bandwidth-bound *decode*.

This module owns the loop instead, so each phase can be built, run and measured
on its own:

* :class:`LLMBenchmarkConfig` -- declarative sweep/model configuration.
* :class:`LLMBenchmark`       -- prefill / decode_step / generate primitives.
* :class:`PagedCache`         -- a block-allocated KV cache (``--cache-type paged``).
* weight/activation quantization for ``w4a16 / w8a16 / w8a8 / fp8``.

It is wired into ``common.py`` through :func:`add_llm_args` and
:func:`run_llm_benchmark`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


CACHE_TYPES = ("static", "paged")
QUANTIZATION_MODES = ("none", "w4a16", "w8a16", "w8a8", "fp8")
SAMPLING_MODES = ("greedy", "top-k", "top-p")
LLM_MODES = ("prefill", "decode", "e2e")

DEFAULT_PROMPT_LENGTHS = (128, 512, 1024, 4096, 8192, 32768)
DEFAULT_DECODE_LENGTHS = (256, 1024, 4096, 16384)

# Peak HBM bandwidth in GB/s, used to turn achieved bandwidth into a utilization
# fraction for the decode benchmark. torch exposes no query for this, so the
# figures come from vendor specs; unknown devices report utilization as None
# rather than guessing.
_PEAK_BANDWIDTH_GBPS = {
    "H100": 3350.0,
    "H200": 4800.0,
    "A100": 2039.0,
    "B200": 8000.0,
    "L40S": 864.0,
    "L4": 300.0,
}

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype(dtype: Any) -> torch.dtype:
    """Accept a torch dtype or one of the common string spellings."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key not in _DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[key]


def peak_bandwidth_gbps(device: torch.device | str | None = None) -> float | None:
    """Best-effort peak HBM bandwidth for ``device``, or None if unknown."""
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(device)
    for key, value in _PEAK_BANDWIDTH_GBPS.items():
        if key.lower() in name.lower():
            return value
    return None


@dataclasses.dataclass
class LLMBenchmarkConfig:
    """Configuration for a decomposed LLM benchmark run.

    ``prompt_lengths`` / ``decode_lengths`` / ``batch_sizes`` describe the sweep;
    the single-point ``--prompt-length`` / ``--decode-length`` CLI flags collapse
    the corresponding list to one entry.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    prompt_lengths: list[int] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_PROMPT_LENGTHS)
    )
    decode_lengths: list[int] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_DECODE_LENGTHS)
    )
    batch_sizes: list[int] = dataclasses.field(default_factory=lambda: [1])
    dtype: Any = torch.bfloat16
    cache_type: str = "static"
    quantization: str = "none"

    device: str = "cuda"
    page_size: int = 128
    sampling: str = "greedy"
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 32
    warmup: int = 2
    iters: int = 5
    compile: bool = False
    # Prefill returns only the final position's logits by default. Full
    # [batch, seq, vocab] logits are ~10 GB per batch element at seq=32768 with
    # this vocabulary, which does not fit alongside the model and KV cache.
    full_logits: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cache_type not in CACHE_TYPES:
            raise ValueError(f"cache_type must be one of {CACHE_TYPES}, got {self.cache_type!r}")
        if self.quantization not in QUANTIZATION_MODES:
            raise ValueError(
                f"quantization must be one of {QUANTIZATION_MODES}, got {self.quantization!r}"
            )
        if self.sampling not in SAMPLING_MODES:
            raise ValueError(f"sampling must be one of {SAMPLING_MODES}, got {self.sampling!r}")
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        self.dtype = resolve_dtype(self.dtype)
        self.prompt_lengths = [int(x) for x in self.prompt_lengths]
        self.decode_lengths = [int(x) for x in self.decode_lengths]
        self.batch_sizes = [int(x) for x in self.batch_sizes]

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["dtype"] = str(self.dtype).replace("torch.", "")
        return out


# ---------------------------------------------------------------------------
# Quantization
#
# Self-contained nn.Linear replacements so the harness can sweep quantization
# modes without pulling in an external quantization library. Weight-only modes
# dequantize into the activation dtype; w8a8 and fp8 use the fused kernels when
# the shapes/device allow and fall back to a dequantized matmul otherwise.
# ---------------------------------------------------------------------------


class _QuantLinearBase(nn.Module):
    mode = "none"

    def __init__(self, in_features: int, out_features: int, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("bias", bias, persistent=False)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode={self.mode}"


class WeightOnlyInt8Linear(_QuantLinearBase):
    """w8a16: symmetric per-output-channel int8 weights, 16-bit activations."""

    mode = "w8a16"

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__(linear.in_features, linear.out_features, linear.bias)
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scale", scale.to(linear.weight.dtype), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, weight, self.bias)


class WeightOnlyInt4Linear(_QuantLinearBase):
    """w4a16: group-wise symmetric int4 weights packed two-per-byte."""

    mode = "w4a16"

    def __init__(self, linear: nn.Linear, group_size: int = 128) -> None:
        super().__init__(linear.in_features, linear.out_features, linear.bias)
        weight = linear.weight.detach().float()
        out_features, in_features = weight.shape
        group_size = math.gcd(group_size, in_features) or in_features
        self.group_size = group_size
        grouped = weight.reshape(out_features, in_features // group_size, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.round(grouped / scale).clamp(-8, 7).to(torch.int8)
        # Pack two signed 4-bit values per byte (low nibble first).
        q_shifted = (q + 8).to(torch.uint8).reshape(out_features, -1)
        packed = (q_shifted[:, 0::2] | (q_shifted[:, 1::2] << 4)).contiguous()
        self.register_buffer("packed", packed, persistent=False)
        self.register_buffer("scale", scale.to(linear.weight.dtype), persistent=False)

    def _dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        low = (self.packed & 0x0F).to(torch.int16) - 8
        high = ((self.packed >> 4) & 0x0F).to(torch.int16) - 8
        interleaved = torch.stack((low, high), dim=-1).reshape(self.out_features, -1)
        grouped = interleaved.reshape(self.out_features, -1, self.group_size).to(dtype)
        return (grouped * self.scale.to(dtype)).reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._dequantize(x.dtype), self.bias)


class Int8DynamicActLinear(_QuantLinearBase):
    """w8a8: int8 weights with per-token dynamic int8 activation quantization."""

    mode = "w8a8"

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__(linear.in_features, linear.out_features, linear.bias)
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scale", scale.squeeze(-1).to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        flat = x.reshape(-1, self.in_features)
        act_scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8).float() / 127.0
        qact = torch.round(flat.float() / act_scale).clamp(-127, 127).to(torch.int8)
        acc = None
        if qact.is_cuda and hasattr(torch, "_int_mm") and qact.shape[0] % 8 == 0:
            try:
                acc = torch._int_mm(qact, self.qweight.t())
            except (RuntimeError, NotImplementedError):
                acc = None
        if acc is None:
            acc = qact.float() @ self.qweight.t().float()
        out = acc.float() * act_scale * self.scale.unsqueeze(0)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(out_dtype).reshape(*x.shape[:-1], self.out_features)


class Fp8Linear(_QuantLinearBase):
    """fp8: per-tensor scaled float8_e4m3 weights, via _scaled_mm when available."""

    mode = "fp8"

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__(linear.in_features, linear.out_features, linear.bias)
        weight = linear.weight.detach().float()
        fp8_dtype = torch.float8_e4m3fn
        finfo_max = torch.finfo(fp8_dtype).max
        scale = weight.abs().amax().clamp(min=1e-8) / finfo_max
        qweight = (weight / scale).clamp(-finfo_max, finfo_max).to(fp8_dtype)
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("weight_scale", scale.reshape(1).to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        flat = x.reshape(-1, self.in_features)
        fp8_dtype = torch.float8_e4m3fn
        finfo_max = torch.finfo(fp8_dtype).max
        act_scale = flat.abs().amax().clamp(min=1e-8).float() / finfo_max
        qact = (flat.float() / act_scale).clamp(-finfo_max, finfo_max).to(fp8_dtype)
        out = None
        # _scaled_mm needs the second operand column-major and K a multiple of 16.
        if qact.is_cuda and hasattr(torch, "_scaled_mm") and self.in_features % 16 == 0:
            try:
                out = torch._scaled_mm(
                    qact,
                    self.qweight.t().contiguous().t(),
                    scale_a=act_scale.reshape(1),
                    scale_b=self.weight_scale,
                    out_dtype=torch.float32,
                )
            except (RuntimeError, NotImplementedError):
                out = None
        if out is None:
            out = (qact.float() @ self.qweight.t().float()) * act_scale * self.weight_scale
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(out_dtype).reshape(*x.shape[:-1], self.out_features)


_QUANT_IMPLS = {
    "w8a16": WeightOnlyInt8Linear,
    "w4a16": WeightOnlyInt4Linear,
    "w8a8": Int8DynamicActLinear,
    "fp8": Fp8Linear,
}


def apply_quantization(model: nn.Module, mode: str, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """Swap ``nn.Linear`` layers in-place for the quantized variant of ``mode``.

    ``lm_head`` is skipped by default: it is a single large projection whose
    output feeds sampling directly, so quantizing it changes generated tokens
    without meaningfully changing the measured phase cost.
    """
    if mode == "none":
        return model
    if mode not in _QUANT_IMPLS:
        raise ValueError(f"quantization must be one of {QUANTIZATION_MODES}, got {mode!r}")
    impl = _QUANT_IMPLS[mode]

    def _convert(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            qualified = f"{prefix}{name}"
            if isinstance(child, nn.Linear) and not any(s in qualified for s in skip):
                setattr(module, name, impl(child).to(child.weight.device))
            else:
                _convert(child, prefix=f"{qualified}.")

    _convert(model)
    return model


# ---------------------------------------------------------------------------
# Paged KV cache
# ---------------------------------------------------------------------------

try:  # pragma: no cover - transformers is a hard dependency of this harness
    from transformers.cache_utils import Cache, CacheLayerMixin, StaticCache
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchbench_llm requires transformers with the layered Cache API"
    ) from exc


class PagedLayer(CacheLayerMixin):
    """One layer of a block-allocated KV cache.

    Keys and values live in a pool of fixed-size pages rather than one
    contiguous per-sequence buffer, with a block table mapping logical page
    index to physical page. Pages are allocated on demand, so a sequence only
    occupies ``ceil(len / page_size)`` pages instead of the full ``max_cache_len``.

    Attention still consumes a contiguous ``[batch, heads, len, dim]`` tensor, so
    :meth:`update` gathers the active pages into one. That keeps the allocator
    honest without requiring a paged-attention kernel.
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, max_cache_len: int, page_size: int = 128) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.max_pages_per_seq = math.ceil(max_cache_len / page_size)
        self.cumulative_length = 0
        self.block_table: torch.Tensor | None = None
        self._next_free_page = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]

        num_pages = self.batch_size * self.max_pages_per_seq
        self.keys = torch.zeros(
            (num_pages, self.num_heads, self.page_size, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (num_pages, self.num_heads, self.page_size, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # -1 marks an unallocated logical page.
        self.block_table = torch.full(
            (self.batch_size, self.max_pages_per_seq), -1, dtype=torch.long, device=self.device
        )
        self._next_free_page = 0
        self.is_initialized = True

    def _ensure_pages(self, upto_length: int) -> None:
        """Allocate physical pages so every sequence covers ``upto_length`` tokens."""
        needed = math.ceil(upto_length / self.page_size)
        if needed > self.max_pages_per_seq:
            raise ValueError(
                f"paged cache overflow: need {needed} pages/sequence but capacity is "
                f"{self.max_pages_per_seq} (max_cache_len={self.max_cache_len})"
            )
        for logical in range(needed):
            if bool((self.block_table[:, logical] >= 0).all()):
                continue
            for batch in range(self.batch_size):
                if int(self.block_table[batch, logical]) < 0:
                    self.block_table[batch, logical] = self._next_free_page
                    self._next_free_page += 1

    def _gather(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize a contiguous view of the first ``length`` cached tokens."""
        num_pages = math.ceil(length / self.page_size)
        pages = self.block_table[:, :num_pages].reshape(-1)

        def _join(pool: torch.Tensor, head_dim: int) -> torch.Tensor:
            # pool[pages] is [B*pages, H, page_size, D]. The page and page-offset
            # axes are separated by H, so they cannot be flattened directly --
            # regroup to [B, pages, H, page_size, D], move H forward, and only
            # then merge (pages, page_size) into one sequence axis.
            gathered = pool[pages].reshape(
                self.batch_size, num_pages, self.num_heads, self.page_size, head_dim
            )
            gathered = gathered.permute(0, 2, 1, 3, 4)
            merged = gathered.reshape(
                self.batch_size, self.num_heads, num_pages * self.page_size, head_dim
            )
            return merged[:, :, :length].contiguous()

        return _join(self.keys, self.k_head_dim), _join(self.values, self.v_head_dim)

    def _scatter(self, key_states: torch.Tensor, value_states: torch.Tensor, start: int) -> None:
        new_len = key_states.shape[-2]
        positions = torch.arange(start, start + new_len, device=self.device)
        logical_pages = positions // self.page_size
        offsets = positions % self.page_size
        for batch in range(self.batch_size):
            physical = self.block_table[batch, logical_pages]
            self.keys[physical, :, offsets] = key_states[batch].transpose(0, 1)
            self.values[physical, :, offsets] = value_states[batch].transpose(0, 1)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        new_len = key_states.shape[-2]
        end = self.cumulative_length + new_len
        self._ensure_pages(end)
        self._scatter(key_states, value_states, self.cumulative_length)
        self.cumulative_length = end
        return self._gather(end)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        if self.is_initialized:
            self.keys.zero_()
            self.values.zero_()
            self.block_table.fill_(-1)
        self.cumulative_length = 0
        self._next_free_page = 0

    def num_allocated_pages(self) -> int:
        """Physical pages handed out so far -- used to show paging actually happens."""
        return self._next_free_page


class PagedCache(Cache):
    """A :class:`Cache` whose layers are block-allocated :class:`PagedLayer` s."""

    def __init__(self, config: Any, max_cache_len: int, page_size: int = 128) -> None:
        text_config = config.get_text_config(decoder=True)
        layers = [
            PagedLayer(max_cache_len=max_cache_len, page_size=page_size)
            for _ in range(text_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.max_cache_len = max_cache_len
        self.page_size = page_size

    def num_allocated_pages(self) -> int:
        return sum(layer.num_allocated_pages() for layer in self.layers)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, exclude_embeddings: bool = True) -> int:
    """Parameter count, optionally excluding embedding/lm_head lookups.

    Quantized layers hold packed buffers rather than Parameters, so their
    logical element count is reconstructed from in/out features.
    """
    total = 0
    embedding_ids: set[int] = set()
    if exclude_embeddings:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embedding_ids.add(id(module.weight))
    for param in model.parameters():
        if id(param) not in embedding_ids:
            total += param.numel()
    for module in model.modules():
        if isinstance(module, _QuantLinearBase):
            total += module.in_features * module.out_features
    return total


def prefill_flops(model_config: Any, num_params: int, batch_size: int, seq_len: int) -> float:
    """Forward FLOPs for a prefill of ``[batch_size, seq_len]``.

    Two terms: dense matmuls (``2 * params * tokens``) and the quadratic
    attention scores (``QK^T`` plus ``AV``), which dominate at long context.
    """
    tokens = batch_size * seq_len
    dense = 2.0 * num_params * tokens
    layers = model_config.num_hidden_layers
    heads = model_config.num_attention_heads
    head_dim = getattr(model_config, "head_dim", None) or (
        model_config.hidden_size // heads
    )
    attention = 4.0 * layers * batch_size * heads * head_dim * (seq_len**2)
    return dense + attention


def kv_cache_bytes(model_config: Any, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    """Bytes occupied by the KV cache for ``seq_len`` tokens."""
    layers = model_config.num_hidden_layers
    kv_heads = getattr(model_config, "num_key_value_heads", None) or (
        model_config.num_attention_heads
    )
    head_dim = getattr(model_config, "head_dim", None) or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    element_size = torch.tensor([], dtype=dtype).element_size()
    return 2.0 * layers * batch_size * kv_heads * head_dim * seq_len * element_size


def model_weight_bytes(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buf in model.buffers():
        total += buf.numel() * buf.element_size()
    return total


def _sync(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _time_calls(fn, warmup: int, iters: int, device: torch.device | str) -> list[float]:
    """Run ``fn`` ``warmup`` then ``iters`` times, returning per-iteration seconds."""
    for _ in range(max(0, warmup)):
        fn()
    _sync(device)
    timings = []
    for _ in range(max(1, iters)):
        start = time.perf_counter()
        fn()
        _sync(device)
        timings.append(time.perf_counter() - start)
    return timings


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


class LLMBenchmark:
    """Owns the generation loop so prefill and decode can be measured apart."""

    def __init__(self, config: LLMBenchmarkConfig | None = None) -> None:
        self.config = config or LLMBenchmarkConfig()
        self.model = None
        self.tokenizer = None

    # -- setup ------------------------------------------------------------
    def setup_model(self, config: LLMBenchmarkConfig | None = None):
        """Load and prepare the model/tokenizer described by ``config``."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = config or self.config
        self.config = config
        torch.manual_seed(config.seed)

        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=config.dtype
        )
        model = model.to(config.device)
        model.eval()
        model.config.use_cache = True
        if config.quantization != "none":
            apply_quantization(model, config.quantization)

        self.model, self.tokenizer = model, tokenizer
        return model, tokenizer

    # -- cache ------------------------------------------------------------
    def make_cache(self, model, max_cache_len: int, batch_size: int):
        """Build the KV cache selected by ``config.cache_type``."""
        if self.config.cache_type == "paged":
            return PagedCache(
                model.config, max_cache_len=max_cache_len, page_size=self.config.page_size
            )
        return StaticCache(config=model.config, max_cache_len=max_cache_len)

    # -- input construction ----------------------------------------------
    def build_prefill_inputs(self, prompt_length: int, batch_size: int) -> torch.Tensor:
        """Random ``[batch_size, prompt_length]`` token ids."""
        vocab_size = self._vocab_size()
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        return torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, prompt_length),
            dtype=torch.long,
            generator=generator,
        ).to(self.config.device)

    def build_decode_inputs(self, cache_depth: int, batch_size: int):
        """A single next token plus a cache already populated to ``cache_depth``.

        The cache is filled directly rather than by running a prefill, so decode
        cost at depth can be measured without paying for the prefill first.
        """
        model = self._require_model()
        cfg = model.config
        device = self.config.device
        dtype = self.config.dtype

        kv_cache = self.make_cache(model, max_cache_len=cache_depth + 1, batch_size=batch_size)
        kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for layer_idx in range(cfg.num_hidden_layers):
            shape = (batch_size, kv_heads, cache_depth, head_dim)
            keys = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                device=device, dtype=dtype
            )
            values = torch.randn(shape, generator=generator, dtype=torch.float32).to(
                device=device, dtype=dtype
            )
            kv_cache.layers[layer_idx].update(keys, values)

        token = torch.randint(
            low=0,
            high=self._vocab_size(),
            size=(batch_size, 1),
            dtype=torch.long,
            generator=generator,
        ).to(device)
        position = torch.tensor([cache_depth], device=device, dtype=torch.long)
        return token, kv_cache, position

    # -- phases -----------------------------------------------------------
    def prefill(self, model, input_ids: torch.Tensor):
        """Run the prompt through the model, returning logits and a filled cache."""
        batch_size, prompt_length = input_ids.shape
        kv_cache = self.make_cache(
            model,
            max_cache_len=prompt_length + self.config.max_new_tokens,
            batch_size=batch_size,
        )
        cache_position = torch.arange(prompt_length, device=input_ids.device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=0 if self.config.full_logits else 1,
            )
        return out.logits, kv_cache

    def decode_step(self, model, token: torch.Tensor, kv_cache, cache_position: torch.Tensor):
        """One autoregressive step against an existing cache."""
        if token.dim() == 1:
            token = token.unsqueeze(-1)
        with torch.no_grad():
            out = model(
                input_ids=token,
                past_key_values=kv_cache,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
            )
        return out.logits, kv_cache

    # -- sampling ---------------------------------------------------------
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Pick the next token per ``config.sampling``."""
        last = logits[:, -1, :].float()
        mode = self.config.sampling
        if mode == "greedy":
            return last.argmax(dim=-1, keepdim=True)
        if mode == "top-k":
            k = min(self.config.top_k, last.shape[-1])
            values, indices = torch.topk(last, k, dim=-1)
            probs = F.softmax(values, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            return indices.gather(-1, choice)
        # top-p (nucleus)
        ordered, indices = torch.sort(last, descending=True, dim=-1)
        probs = F.softmax(ordered, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        keep = cumulative - probs <= self.config.top_p
        keep[..., 0] = True
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        probs = probs / probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, choice)

    # -- the loop ---------------------------------------------------------
    def generate(self, model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Prefill once, then decode ``max_new_tokens`` steps, owning the loop.

        The loop is a fixed trip count over static shapes with no data-dependent
        control flow, so the whole method compiles rather than only
        ``model.forward``.
        """
        logits, kv_cache = self.prefill(model, input_ids)
        next_token = self.sample(logits)
        generated = [next_token]

        prompt_length = input_ids.shape[1]
        for step in range(max_new_tokens - 1):
            cache_position = torch.tensor(
                [prompt_length + step], device=input_ids.device, dtype=torch.long
            )
            logits, kv_cache = self.decode_step(model, next_token, kv_cache, cache_position)
            next_token = self.sample(logits)
            generated.append(next_token)

        return torch.cat([input_ids] + generated, dim=1)

    def compiled_generate(self):
        """``generate`` compiled as a whole, not just ``model.forward``."""
        return torch.compile(self.generate, dynamic=False)

    # -- benchmarks -------------------------------------------------------
    def benchmark_prefill(self, prompt_length: int, batch_size: int) -> dict[str, Any]:
        """TTFT and TFLOPS for a single prefill point."""
        model = self._require_model()
        input_ids = self.build_prefill_inputs(prompt_length, batch_size)
        num_params = count_parameters(model)

        def run() -> None:
            self.prefill(model, input_ids)

        timings = _time_calls(run, self.config.warmup, self.config.iters, self.config.device)
        seconds = _median(timings)
        flops = prefill_flops(model.config, num_params, batch_size, prompt_length)
        return {
            "phase": "prefill",
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "ttft_ms": seconds * 1e3,
            "tflops": flops / seconds / 1e12,
            "cache_type": self.config.cache_type,
            "quantization": self.config.quantization,
        }

    def benchmark_decode(self, cache_depth: int, batch_size: int) -> dict[str, Any]:
        """Per-token latency and memory-bandwidth utilization at a cache depth."""
        model = self._require_model()
        token, kv_cache, position = self.build_decode_inputs(cache_depth, batch_size)

        def run() -> None:
            self.decode_step(model, token, kv_cache, position)

        timings = _time_calls(run, self.config.warmup, self.config.iters, self.config.device)
        seconds = _median(timings)

        # A decode step is bandwidth-bound: it streams the weights once and
        # re-reads the whole KV cache.
        moved = model_weight_bytes(model) + kv_cache_bytes(
            model.config, batch_size, cache_depth, self.config.dtype
        )
        achieved = moved / seconds / 1e9
        peak = peak_bandwidth_gbps(self.config.device)
        result = {
            "phase": "decode",
            "batch_size": batch_size,
            "cache_depth": cache_depth,
            "latency_per_token_ms": seconds * 1e3,
            "achieved_bandwidth_gbps": achieved,
            "peak_bandwidth_gbps": peak,
            "memory_bandwidth_utilization": (achieved / peak) if peak else None,
            "cache_type": self.config.cache_type,
            "quantization": self.config.quantization,
        }
        if isinstance(kv_cache, PagedCache):
            result["allocated_pages"] = kv_cache.num_allocated_pages()
        return result

    def benchmark_e2e(self, prompt_length: int, batch_size: int, max_new_tokens: int | None = None):
        """End-to-end generation, instrumented per phase."""
        model = self._require_model()
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        input_ids = self.build_prefill_inputs(prompt_length, batch_size)
        num_params = count_parameters(model)

        _sync(self.config.device)
        start = time.perf_counter()
        logits, kv_cache = self.prefill(model, input_ids)
        _sync(self.config.device)
        prefill_seconds = time.perf_counter() - start

        next_token = self.sample(logits)
        decode_seconds = 0.0
        for step in range(max_new_tokens - 1):
            cache_position = torch.tensor(
                [prompt_length + step], device=input_ids.device, dtype=torch.long
            )
            _sync(self.config.device)
            step_start = time.perf_counter()
            logits, kv_cache = self.decode_step(model, next_token, kv_cache, cache_position)
            _sync(self.config.device)
            decode_seconds += time.perf_counter() - step_start
            next_token = self.sample(logits)

        decode_steps = max(1, max_new_tokens - 1)
        total = prefill_seconds + decode_seconds
        flops = prefill_flops(model.config, num_params, batch_size, prompt_length)
        return {
            "phase": "e2e",
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "max_new_tokens": max_new_tokens,
            "ttft_ms": prefill_seconds * 1e3,
            "tflops": flops / prefill_seconds / 1e12,
            "decode_total_ms": decode_seconds * 1e3,
            "latency_per_token_ms": decode_seconds / decode_steps * 1e3,
            "total_ms": total * 1e3,
            "throughput_tokens_per_s": max_new_tokens / total,
            "cache_type": self.config.cache_type,
            "quantization": self.config.quantization,
        }

    # -- sweeps -----------------------------------------------------------
    def run_sweep(self, mode: str) -> list[dict[str, Any]]:
        """Run ``mode`` across the configured sweep axes."""
        if mode not in LLM_MODES:
            raise ValueError(f"llm mode must be one of {LLM_MODES}, got {mode!r}")
        if self.model is None:
            self.setup_model(self.config)
        results = []
        for batch_size in self.config.batch_sizes:
            if mode == "prefill":
                for prompt_length in self.config.prompt_lengths:
                    results.append(self.benchmark_prefill(prompt_length, batch_size))
            elif mode == "decode":
                for cache_depth in self.config.decode_lengths:
                    results.append(self.benchmark_decode(cache_depth, batch_size))
            else:
                for prompt_length in self.config.prompt_lengths:
                    results.append(self.benchmark_e2e(prompt_length, batch_size))
        return results

    # -- helpers ----------------------------------------------------------
    def _require_model(self):
        if self.model is None:
            self.setup_model(self.config)
        return self.model

    def _vocab_size(self) -> int:
        if self.model is not None:
            return int(self.model.config.vocab_size)
        from transformers import AutoConfig

        return int(AutoConfig.from_pretrained(self.config.model_name).vocab_size)


# ---------------------------------------------------------------------------
# CLI integration with common.py
# ---------------------------------------------------------------------------


def add_llm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the decomposed-LLM flags on the dynamo benchmark parser."""
    group = parser.add_argument_group("decomposed LLM benchmarks")
    group.add_argument(
        "--llm-mode",
        choices=LLM_MODES,
        default=None,
        help="Run the decomposed LLM harness in prefill-only, decode-only, or end-to-end mode",
    )
    group.add_argument(
        "--prompt-length",
        type=int,
        default=None,
        help="Single prompt length for --llm-mode (default: sweep)",
    )
    group.add_argument(
        "--decode-length",
        type=int,
        default=None,
        help="Single KV-cache depth for --llm-mode decode (default: sweep)",
    )
    group.add_argument(
        "--cache-type",
        choices=CACHE_TYPES,
        default="static",
        help="KV cache implementation for --llm-mode",
    )
    group.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-0.6B",
        help="Model to benchmark with --llm-mode",
    )
    group.add_argument(
        "--llm-quantization",
        choices=QUANTIZATION_MODES,
        default="none",
        help="Quantization scheme for --llm-mode",
    )
    group.add_argument(
        "--llm-batch-size", type=int, default=1, help="Batch size for --llm-mode"
    )
    group.add_argument(
        "--llm-max-new-tokens", type=int, default=32, help="Tokens to generate for --llm-mode e2e"
    )
    return parser


def config_from_args(args: Any) -> LLMBenchmarkConfig:
    """Build an :class:`LLMBenchmarkConfig` from parsed CLI args."""
    prompt_lengths = (
        [args.prompt_length] if getattr(args, "prompt_length", None) else list(DEFAULT_PROMPT_LENGTHS)
    )
    decode_lengths = (
        [args.decode_length] if getattr(args, "decode_length", None) else list(DEFAULT_DECODE_LENGTHS)
    )
    dtype = torch.float16 if getattr(args, "float16", False) else torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LLMBenchmarkConfig(
        model_name=getattr(args, "llm_model", "Qwen/Qwen3-0.6B"),
        prompt_lengths=prompt_lengths,
        decode_lengths=decode_lengths,
        batch_sizes=[getattr(args, "llm_batch_size", 1)],
        dtype=dtype,
        cache_type=getattr(args, "cache_type", "static") or "static",
        quantization=getattr(args, "llm_quantization", "none") or "none",
        device=device,
        max_new_tokens=getattr(args, "llm_max_new_tokens", 32),
    )


def run_llm_benchmark(args: Any) -> list[dict[str, Any]]:
    """Entry point used by ``common.main`` when ``--llm-mode`` is supplied."""
    config = config_from_args(args)
    benchmark = LLMBenchmark(config)
    benchmark.setup_model(config)
    results = benchmark.run_sweep(args.llm_mode)
    for row in results:
        print(json.dumps(row, sort_keys=True))
    return results

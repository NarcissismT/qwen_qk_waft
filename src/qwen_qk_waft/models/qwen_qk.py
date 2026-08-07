"""Frozen Qwen-Image-Edit target-Q/source-K feature extraction."""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor, nn


_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
    "img_mlp.net.2",
    "img_mod.1",
    "txt_mlp.net.2",
    "txt_mod.1",
)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }[name.lower()]


@dataclass(frozen=True)
class TokenLayout:
    target_grid: tuple[int, int]
    source_grid: tuple[int, int]
    target_tokens: int
    source_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.target_tokens + self.source_tokens

    @classmethod
    def from_img_shapes(cls, value: Any) -> "TokenLayout":
        shapes = value
        if len(shapes) == 1 and isinstance(shapes[0][0], (list, tuple)):
            shapes = shapes[0]
        target, source = shapes
        target_frames, target_h, target_w = (int(v) for v in target)
        source_frames, source_h, source_w = (int(v) for v in source)
        return cls(
            target_grid=(target_h, target_w),
            source_grid=(source_h, source_w),
            target_tokens=target_frames * target_h * target_w,
            source_tokens=source_frames * source_h * source_w,
        )

    def target(self, value: Tensor) -> Tensor:
        return value[:, : self.target_tokens]

    def source(self, value: Tensor) -> Tensor:
        return value[:, self.target_tokens : self.total_tokens]


@dataclass(frozen=True)
class FeaturePacket:
    step: int
    layer: int
    variant: str
    target: Tensor
    source: Tensor
    target_grid: tuple[int, int]
    source_grid: tuple[int, int]


def apply_rotary(value: Tensor, frequencies: Tensor) -> Tensor:
    complex_value = torch.view_as_complex(
        value.float().reshape(*value.shape[:-1], value.shape[-1] // 2, 2)
    )
    frequencies = frequencies.to(value.device)
    if not torch.is_complex(frequencies):
        frequencies = torch.view_as_complex(
            frequencies.float().reshape(frequencies.shape[0], -1, 2)
        )
    rotated = complex_value * frequencies.unsqueeze(1)
    return torch.view_as_real(rotated).flatten(-2).to(value.dtype)


def _standardize_qk(value: Tensor, token_count: int, heads: int) -> Tensor:
    if value.ndim == 4 and value.shape[1] == token_count:
        return value
    if value.ndim == 4 and value.shape[2] == token_count:
        return value.transpose(1, 2)
    if value.ndim == 3:
        return value.unflatten(-1, (heads, value.shape[-1] // heads))
    raise ValueError(f"unsupported Q/K shape {tuple(value.shape)}")


def _load_lora(transformer: nn.Module, checkpoint: str | Path, scale: float) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import load_file

    raw = load_file(str(checkpoint), device="cpu")
    state: dict[str, Tensor] = {}
    ranks = set()
    for raw_key, value in raw.items():
        key = raw_key
        for prefix in ("module.", "pipe.dit.", "dit.", "transformer."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        state[key] = value
        if ".lora_A." in key:
            ranks.add(int(value.shape[0]))
    rank = ranks.pop()
    config = LoraConfig(
        r=rank,
        lora_alpha=float(rank) * float(scale),
        target_modules=list(_LORA_TARGETS),
        bias="none",
    )
    transformer = inject_adapter_in_model(config, transformer)
    transformer.load_state_dict(state, strict=False)
    return transformer


class QwenQKExtractor:
    """Run the fixed target-latent trajectory and emit selected descriptors.

    The real rectified image is never accepted by this API.  Token segments
    are derived from every transformer call's runtime ``img_shapes`` metadata.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: torch.device,
        layers: Sequence[int],
        steps: Sequence[int],
        variants: Sequence[str] = ("pre", "post", "hidden"),
    ) -> None:
        self.config = dict(config)
        self.device = torch.device(device)
        self.layers = tuple(int(v) for v in layers)
        self.steps = frozenset(int(v) for v in steps)
        self.variants = frozenset(str(v) for v in variants)
        self.pipeline = self._load_pipeline()
        block_count = len(self.pipeline.transformer.transformer_blocks)
        self.layers = tuple(v if v >= 0 else block_count + v for v in self.layers)
        self._handles: list[Any] = []
        self._layout: TokenLayout | None = None
        self._frequencies: Tensor | None = None
        self._queries: dict[int, Tensor] = {}
        self._step = 0
        self._calls = 0
        self._active = False
        self._consumer: Callable[[FeaturePacket], None] | None = None
        self._register_hooks()

    def _load_pipeline(self) -> Any:
        import diffusers

        pipeline_name = str(self.config.get("pipeline_class", "QwenImageEditPipeline"))
        pipeline_type = getattr(diffusers, pipeline_name)
        model_root = str(self.config["model_root"])
        dtype = _dtype(str(self.config.get("dtype", "bfloat16")))
        pipeline = pipeline_type.from_pretrained(
            model_root,
            torch_dtype=dtype,
            local_files_only=bool(self.config.get("local_files_only", True)),
        )
        lora_scale = float(self.config.get("lora_scale", 1.0))
        lora_checkpoint = self.config.get("lora_checkpoint")
        if lora_checkpoint and lora_scale > 0:
            pipeline.transformer = _load_lora(
                pipeline.transformer, str(lora_checkpoint), lora_scale
            )
        if bool(self.config.get("cpu_offload", False)):
            pipeline.enable_model_cpu_offload(gpu_id=self.device.index or 0)
        else:
            pipeline.to(self.device)
        for name in ("transformer", "vae", "text_encoder"):
            module = getattr(pipeline, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def _register_hooks(self) -> None:
        transformer = self.pipeline.transformer
        self._handles.append(
            transformer.register_forward_pre_hook(self._transformer_pre, with_kwargs=True)
        )
        self._handles.append(transformer.pos_embed.register_forward_hook(self._rope_hook))
        for layer in self.layers:
            block = transformer.transformer_blocks[layer]
            attention = block.attn
            self._handles.append(
                attention.norm_q.register_forward_hook(
                    self._query_hook(layer, int(attention.heads))
                )
            )
            self._handles.append(
                attention.norm_k.register_forward_hook(
                    self._key_hook(layer, int(attention.heads))
                )
            )
            self._handles.append(block.register_forward_hook(self._hidden_hook(layer)))

    def _transformer_pre(
        self, _module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        conditional = self._calls == 0
        self._calls += 1
        self._active = conditional and self._step in self.steps
        self._queries.clear()
        self._frequencies = None
        self._layout = TokenLayout.from_img_shapes(kwargs["img_shapes"])

    def _rope_hook(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if self._active:
            self._frequencies = output[0]

    def _query_hook(self, layer: int, heads: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
            if self._active and self._layout is not None:
                query = _standardize_qk(output, self._layout.total_tokens, heads)
                self._queries[layer] = self._layout.target(query)

        return hook

    def _emit(
        self,
        layer: int,
        variant: str,
        target: Tensor,
        source: Tensor,
    ) -> None:
        if self._consumer is None or self._layout is None:
            return
        target = target.reshape(target.shape[0], target.shape[1], -1).detach().to(self.device)
        source = source.reshape(source.shape[0], source.shape[1], -1).detach().to(self.device)
        self._consumer(
            FeaturePacket(
                step=self._step,
                layer=layer,
                variant=variant,
                target=target,
                source=source,
                target_grid=self._layout.target_grid,
                source_grid=self._layout.source_grid,
            )
        )

    def _key_hook(self, layer: int, heads: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
            if not self._active or self._layout is None:
                return
            query = self._queries.pop(layer)
            key = _standardize_qk(output, self._layout.total_tokens, heads)
            source = self._layout.source(key)
            if "pre" in self.variants:
                self._emit(layer, "pre", query, source)
            if "post" in self.variants:
                target_freq = self._frequencies[: self._layout.target_tokens]
                source_freq = self._frequencies[
                    self._layout.target_tokens : self._layout.total_tokens
                ]
                self._emit(
                    layer,
                    "post",
                    apply_rotary(query, target_freq),
                    apply_rotary(source, source_freq),
                )

        return hook

    def _hidden_hook(self, layer: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not self._active or "hidden" not in self.variants or self._layout is None:
                return
            hidden = output[1]
            self._emit(
                layer,
                "hidden",
                self._layout.target(hidden),
                self._layout.source(hidden),
            )

        return hook

    def _step_callback(
        self,
        _pipeline: Any,
        step: int,
        _timestep: Tensor,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self._step = int(step) + 1
        self._calls = 0
        self._active = False
        self._queries.clear()
        return callback_kwargs

    def run(
        self,
        image: Image.Image,
        *,
        seed: int,
        consumer: Callable[[FeaturePacket], None] | None = None,
        store: bool = True,
    ) -> dict[tuple[int, int, str], FeaturePacket]:
        captured: dict[tuple[int, int, str], FeaturePacket] = {}

        def collect(packet: FeaturePacket) -> None:
            if store:
                captured[(packet.step, packet.layer, packet.variant)] = packet
            if consumer is not None:
                consumer(packet)

        self._consumer = collect
        self._step = 0
        self._calls = 0
        execution_device = getattr(self.pipeline, "_execution_device", self.device)
        generator_device = execution_device if str(execution_device).startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        height = int(self.config.get("height", 512))
        width = int(self.config.get("width", 512))
        kwargs: dict[str, Any] = {
            "image": image.convert("RGB"),
            "prompt": str(self.config["prompt"]),
            "height": height,
            "width": width,
            "generator": generator,
            "num_inference_steps": int(self.config.get("num_inference_steps", 4)),
            "true_cfg_scale": float(self.config.get("true_cfg_scale", 1.0)),
            "guidance_scale": float(self.config.get("guidance_scale", 1.0)),
            "num_images_per_prompt": 1,
            "output_type": "latent",
            "callback_on_step_end": self._step_callback,
            "callback_on_step_end_tensor_inputs": ["latents"],
        }
        signature = inspect.signature(self.pipeline.__call__).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in signature}
        try:
            with torch.inference_mode():
                self.pipeline(**kwargs)
        finally:
            self._consumer = None
            self._queries.clear()
        return captured

    def selected_pair(
        self,
        image: Image.Image,
        *,
        seed: int,
        step: int,
        variant: str,
    ) -> tuple[Tensor, Tensor, tuple[int, int], tuple[int, int]]:
        requested = ("pre", "post") if variant == "pre_post" else (variant,)
        old_variants = self.variants
        self.variants = frozenset(requested)
        captured = self.run(image, seed=seed)
        target_layers, source_layers = [], []
        target_grid = source_grid = None
        for layer in self.layers:
            if variant == "pre_post":
                pre = captured[(step, layer, "pre")]
                post = captured[(step, layer, "post")]
                target = torch.cat((pre.target, post.target), dim=-1)
                source = torch.cat((pre.source, post.source), dim=-1)
                packet = pre
            else:
                packet = captured[(step, layer, variant)]
                target, source = packet.target, packet.source
            target_layers.append(
                target.transpose(1, 2).reshape(
                    target.shape[0], target.shape[2], *packet.target_grid
                )
            )
            source_layers.append(
                source.transpose(1, 2).reshape(
                    source.shape[0], source.shape[2], *packet.source_grid
                )
            )
            target_grid, source_grid = packet.target_grid, packet.source_grid
        self.variants = old_variants
        return (
            torch.stack(target_layers, dim=1).clone(),
            torch.stack(source_layers, dim=1).clone(),
            target_grid,
            source_grid,
        )

    def close(self) -> None:
        for handle in self._handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self._handles.clear()

    def __enter__(self) -> "QwenQKExtractor":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

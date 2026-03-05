#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from timesformer.models.vit import VisionTransformer


def build_model(model_params: Dict, device: torch.device) -> torch.nn.Module:
    model = VisionTransformer(
        img_size=model_params["image_size"],
        num_classes=model_params["num_classes"],
        patch_size=model_params["patch_size"],
        embed_dim=model_params["dim"],
        depth=model_params["depth"],
        num_heads=model_params["heads"],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=model_params.get("attn_dropout", 0.0),
        drop_path_rate=model_params.get("ff_dropout", 0.1),
        num_frames=model_params["num_frames"],
        attention_type=model_params["attention_type"],
    )
    return model.to(device).eval()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    checkpoint_name: Optional[str],
    device: torch.device,
) -> None:
    if not checkpoint_path or not checkpoint_name:
        return
    checkpoint_file = os.path.join(checkpoint_path, f"{checkpoint_name}.pth")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    warmup: int,
    measure_iters: int,
    batch_size: int,
    seq_len: int,
    height: int,
    width: int,
) -> Dict[str, float]:
    shape = (batch_size, 3, seq_len, height, width)

    def one_step() -> None:
        x = torch.randn(*shape, device=device)
        y = model(x)
        _ = y.detach().to("cpu", non_blocking=False)

    for _ in range(warmup):
        one_step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(measure_iters):
        one_step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - t0
    total_frames = batch_size * seq_len * measure_iters
    total_clips = batch_size * measure_iters

    return {
        "elapsed_sec": elapsed,
        "total_frames": float(total_frames),
        "total_clips": float(total_clips),
        "fps": total_frames / elapsed,
        "cps": total_clips / elapsed,
        "latency_ms": (elapsed / measure_iters) * 1000.0,
    }


def get_model_params(args: argparse.Namespace) -> Dict:
    if args.checkpoint_path:
        with open(os.path.join(args.checkpoint_path, "args.pkl"), "rb") as f:
            train_args = pickle.load(f)
        model_params = train_args["model_params"]
        return model_params

    return {
        "dim": args.dim,
        "image_size": (args.height, args.width),
        "patch_size": args.patch_size,
        "attention_type": args.attention_type,
        "num_frames": args.seq_len,
        "num_classes": 6 * (args.seq_len - 1),
        "depth": args.depth,
        "heads": args.heads,
        "attn_dropout": args.attn_dropout,
        "ff_dropout": args.ff_dropout,
    }


def resolve_runtime_shape(args: argparse.Namespace, model_params: Dict) -> Tuple[int, int, int]:
    """Return (seq_len, height, width) used for random input tensors.

    If checkpoint config is provided, enforce the exact training-time shape to avoid
    accidental mismatches between CLI values and model weights.
    """
    if args.checkpoint_path:
        h, w = model_params["image_size"]
        t = model_params["num_frames"]
        return t, h, w
    return args.seq_len, args.height, args.width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSformer-VO inference benchmark with fixed objective conditions"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup forward iterations (not timed)")
    parser.add_argument("--measure-iters", type=int, default=200, help="Timed forward iterations")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=1)

    parser.add_argument("--checkpoint-path", type=str, default=None, help="Folder containing args.pkl and checkpoint")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint stem without .pth")

    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--attention-type", type=str, default="divided_space_time")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--ff-dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")

    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)

    torch.backends.cudnn.benchmark = False

    if args.checkpoint_name and not args.checkpoint_path:
        raise ValueError("--checkpoint-name requires --checkpoint-path")

    model_params = get_model_params(args)
    if not args.checkpoint_path:
        model_params["num_frames"] = args.seq_len
        model_params["image_size"] = (args.height, args.width)
        model_params["num_classes"] = 6 * (args.seq_len - 1)

    seq_len, height, width = resolve_runtime_shape(args, model_params)

    model = build_model(model_params, device)
    load_checkpoint(model, args.checkpoint_path, args.checkpoint_name, device)

    print("=" * 72)
    print(f"Device: {device} | AMP: disabled | Warmup: {args.warmup} | Timed iters: {args.measure_iters}")
    print(
        f"Input shape per step: (N={args.batch_size}, C=3, T={seq_len}, H={height}, W={width})"
    )
    if args.checkpoint_path:
        print("Input shape source: loaded from checkpoint args.pkl (overrides CLI --seq-len/--height/--width)")
    if device.type == "cpu":
        print(f"CPU threads: intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}")
    print("Timing scope: tensor -> forward -> tensor(cpu)")
    print("GPU sync policy: synchronize before and after timed region")
    print("=" * 72)

    stats = benchmark_model(
        model=model,
        device=device,
        warmup=args.warmup,
        measure_iters=args.measure_iters,
        batch_size=args.batch_size,
        seq_len=seq_len,
        height=height,
        width=width,
    )

    print("[TSformer-VO]")
    print(f"  elapsed: {stats['elapsed_sec']:.4f} s")
    print(f"  total_clips: {int(stats['total_clips'])}")
    print(f"  total_frames: {int(stats['total_frames'])}")
    print(f"  latency: {stats['latency_ms']:.3f} ms/iter")
    print(f"  CPS (clips/s): {stats['cps']:.3f}")
    print(f"  FPS (frames/s): {stats['fps']:.3f}")
    print("-" * 72)


if __name__ == "__main__":
    main()

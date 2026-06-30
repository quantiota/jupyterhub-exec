#!/usr/bin/env python3
"""
gpu_demo.py — small GPU workload, meant to run on the remote JupyterHub kernel.

Run it on the GPU box via the offload harness:
    /opt/venv/bin/python3 jh_exec.py gpu_demo.py

It reports the visible GPU(s) and runs a matrix multiply on the device to
prove compute actually lands on the GPU.
"""

import torch


def main():
    if not torch.cuda.is_available():
        print("No CUDA device visible to this kernel — running on CPU only.")
        return

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    print(f"GPU: {name}  ({total_gb:.1f} GiB)  torch {torch.__version__}")

    # A matmul big enough to be obviously GPU-bound.
    n = 8192
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = a @ b
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    flops = 2 * n**3
    tflops = flops / (ms / 1000) / 1e12
    print(f"{n}x{n} matmul: {ms:.1f} ms  ({tflops:.1f} TFLOP/s)")
    print(f"checksum: {c.sum().item():.4f}")
    print(f"allocated: {torch.cuda.memory_allocated(dev) / 1024**2:.0f} MiB")


if __name__ == "__main__":
    main()

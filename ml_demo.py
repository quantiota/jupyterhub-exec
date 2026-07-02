#!/usr/bin/env python3
"""
ml_demo.py — the SAME ML workload, run two ways, to show GPU offload with jupyterhub-exec.

    CPU  (locally, in the code-server terminal):   python3 ml_demo.py
    GPU  (on the remote JupyterHub kernel):         jh-exec run ml_demo.py

It trains a small multi-layer perceptron on synthetic data and reports the device it ran
on plus the throughput — so the two runs make the offload obvious side by side.

Scale STEPS / N below if the CPU run is too long (or too short) for your video.
"""
import os
import time
import torch
import torch.nn as nn

# ---- pick the device this run landed on -------------------------------------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
vcpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
print("=" * 52)
print(f"  Device : {name}")
print(f"  torch  : {torch.__version__}")
print(f"  vCPUs  : {vcpus}")
print("=" * 52)

# ---- a small-but-not-trivial training workload ------------------------------
torch.manual_seed(0)
N, D_in, H, D_out = 2048, 2048, 2048, 512   # batch / in / hidden / out
STEPS = 30

x = torch.randn(N, D_in, device=dev)
y = torch.randn(N, D_out, device=dev)

model = nn.Sequential(
    nn.Linear(D_in, H), nn.ReLU(),
    nn.Linear(H, H),    nn.ReLU(),
    nn.Linear(H, D_out),
).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# warm up (allocations / cudnn autotune) so the timed loop is clean
model(x[:64])
if dev.type == "cuda":
    torch.cuda.synchronize()

# ---- timed training loop ----------------------------------------------------
t0 = time.time()
for step in range(STEPS):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    if step == 0 or (step + 1) % 10 == 0:
        print(f"  step {step + 1:>3}/{STEPS}   loss {loss.item():.4f}")
if dev.type == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0

print("-" * 52)
print(f"  {STEPS} steps in {dt:.2f}s   ({STEPS / dt:.1f} steps/s)")
if dev.type == "cuda":
    print(f"  GPU memory used: {torch.cuda.max_memory_allocated() // 1024**2} MiB")
print("=" * 52)
print(f">>> ran on {name} in {dt:.2f}s  ({STEPS / dt:.1f} steps/s)")

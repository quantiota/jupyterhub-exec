# jupyterhub-exec


Born from an [AI agent farm](https://github.com/quantiota/AI-Agent-Farm), solving a real infrastructure problem. Execute code on a remote JupyterHub kernel from any terminal — zero external dependencies.

```
pip install jupyterhub-exec
```

## Why

JupyterHub provides GPU compute. Your agent terminal does not.
`jh-exec` bridges the two using the Jupyter kernel protocol over a raw WebSocket —
no browser, no notebook UI, no library dependencies beyond the Python standard library.

```
┌─────────────────────────┐        WebSocket         ┌──────────────────────────┐
│   Agent Terminal (CPU)  │ ───────────────────────► │  JupyterHub Kernel (GPU) │
│   Claude Code / CLI     │ ◄─────────────────────── │  PyTorch / CUDA          │
└─────────────────────────┘        stdout stream      └──────────────────────────┘
```

## Usage

```bash
# Execute a script on the remote GPU kernel
jh-exec run train.py

# Execute inline code
jh-exec exec "import torch; print(torch.cuda.is_available())"

# List running kernels
jh-exec kernels

# Start a new kernel
jh-exec new-kernel
```

## Configuration

Set via environment variables or a `.env` file in the working or home directory:

**Public JupyterHub (HTTPS — default):**
```bash
JH_HOST=hub.example.com
JH_PORT=443
JH_USER=agent-01
JH_TOKEN=your_token_here
JH_TIMEOUT=600
```

**Local JupyterHub (HTTP):**
```bash
JH_HOST=192.168.1.100
JH_PORT=8000
JH_USER=agent-01
JH_TOKEN=your_token_here
JH_SSL=false
JH_TIMEOUT=600
```

Or pass directly:

```bash
jh-exec --host hub.example.com --port 443 --ssl --user agent-01 --token your_token run script.py
```


## Python API

```python
from jh_exec import execute, list_kernels, new_kernel

# Execute code, stream output to stdout
execute("import torch; print(torch.cuda.get_device_name(0))")

# List running kernels
kernels = list_kernels()

# Start a new kernel, get its ID
kid = new_kernel()
```

## Dedicated GPU per agent

In `jupyterhub_config.py`:

```python
def assign_gpu(spawner):
    gpu_map = {
        "agent-01": "0",
        "agent-02": "1",
        "agent-03": "2",
    }
    spawner.environment["CUDA_VISIBLE_DEVICES"] = gpu_map.get(spawner.user.name, "")

c.Spawner.pre_spawn_hook = assign_gpu
```




## Benchmark

Validated on NVIDIA GeForce GTX TITAN X via `gpu_demo.py`:

```
Visible GPUs: 4  torch 2.5.1+cu121
  [0] NVIDIA GeForce GTX TITAN X  (11.9 GiB)
  [1] NVIDIA GeForce GTX TITAN X  (11.9 GiB)
  [2] NVIDIA GeForce GTX TITAN X  (11.9 GiB)
  [3] NVIDIA GeForce GTX TITAN X  (11.9 GiB)
8192x8192 matmul: 242.1 ms  (4.5 TFLOP/s)
checksum: -489134.2500
allocated: 1720 MiB
```


#### CPU vs GPU — same script, offloaded (`ml_demo.py`)

The same MLP-training script, run locally on CPU vs offloaded to the remote GPU kernel — only
the command changes.

Local, on CPU:

```
$ python3 ml_demo.py
====================================================
  Device : CPU
  torch  : 2.11.0+cu130
  vCPUs  : 8
====================================================
  step   1/30   loss 1.0095
  step  10/30   loss 0.9030
  step  20/30   loss 0.7048
  step  30/30   loss 0.5562
----------------------------------------------------
  30 steps in 35.38s   (0.8 steps/s)
====================================================
>>> ran on CPU in 35.38s  (0.8 steps/s)
```

Offloaded to the remote GPU kernel:

```
$ jh-exec run ml_demo.py
====================================================
  Device : NVIDIA GeForce GTX TITAN X
  torch  : 2.5.1+cu121
  vCPUs  : 12
====================================================
  step   1/30   loss 1.0114
  step  10/30   loss 0.8923
  step  20/30   loss 0.6796
  step  30/30   loss 0.4982
----------------------------------------------------
  30 steps in 0.77s   (39.0 steps/s)
  GPU memory used: 1720 MiB
====================================================
>>> ran on NVIDIA GeForce GTX TITAN X in 0.77s  (39.0 steps/s)
```

**~46× faster**, zero code change, zero local GPU — a CPU-only terminal offloading real
training to a remote GPU it doesn't have.

Full GPU offload from a Claude Code terminal — zero local GPU, zero dependencies.


## License

MIT
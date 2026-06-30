"""
jh_exec — Execute code on a remote JupyterHub kernel from any terminal.
"""

__version__ = "0.1.0"

from .client import execute, list_kernels, new_kernel, get_or_create_kernel

__all__ = ["execute", "list_kernels", "new_kernel", "get_or_create_kernel"]

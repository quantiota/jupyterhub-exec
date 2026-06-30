"""
Command-line interface for jh_exec.
"""

import argparse, sys, os, pathlib
from .client import JupyterHubClient
from . import __version__


def load_env(env_file=None):
    candidates = []
    if env_file:
        candidates.append(pathlib.Path(env_file))
    candidates += [pathlib.Path.cwd() / ".env", pathlib.Path.home() / ".env"]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


def build_client(args):
    return JupyterHubClient(
        host    = args.host    or os.getenv("JH_HOST",    "localhost"),
        port    = args.port    or os.getenv("JH_PORT",    "8000"),
        user    = args.user    or os.getenv("JH_USER",    ""),
        token   = args.token   or os.getenv("JH_TOKEN",   ""),
        timeout = args.timeout or int(os.getenv("JH_TIMEOUT", "600")),
    )


def main():
    parser = argparse.ArgumentParser(
        prog="jh-exec",
        description="Execute code on a remote JupyterHub kernel from any terminal."
    )
    parser.add_argument("--version", action="version", version=f"jh-exec {__version__}")
    parser.add_argument("--host",    help="JupyterHub host")
    parser.add_argument("--port",    help="JupyterHub port (default: 8000)")
    parser.add_argument("--user",    help="JupyterHub username")
    parser.add_argument("--token",   help="JupyterHub API token")
    parser.add_argument("--timeout", type=int, help="Execution timeout in seconds (default: 600)")
    parser.add_argument("--kernel",  help="Kernel ID (auto-discovered if omitted)")
    parser.add_argument("--env",     help="Path to .env file")

    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Execute a Python script file")
    run_p.add_argument("script", help="Path to the Python script")

    # exec
    exec_p = sub.add_parser("exec", help="Execute inline Python code")
    exec_p.add_argument("code", help="Python code to execute")

    # kernels
    sub.add_parser("kernels", help="List running kernels")

    # new-kernel
    sub.add_parser("new-kernel", help="Start a new kernel and print its ID")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    load_env(args.env)
    client = build_client(args)

    if args.command == "kernels":
        kernels = client.list_kernels()
        if not kernels:
            print("No running kernels.")
        for k in kernels:
            print(f"{k['id']}  name={k['name']}  state={k['execution_state']}  last_activity={k['last_activity']}")

    elif args.command == "new-kernel":
        kid = client.new_kernel()
        print(kid)

    elif args.command in ("run", "exec"):
        if args.command == "run":
            with open(args.script) as f:
                code = f.read()
        else:
            code = args.code

        kernel_id = args.kernel or client.get_or_create_kernel()
        status    = client.execute(code, kernel_id=kernel_id)
        sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()

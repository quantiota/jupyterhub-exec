"""
Core WebSocket client for JupyterHub kernel execution.
Zero external dependencies — uses only Python built-ins.
"""

import socket, base64, struct, json, uuid, time, os, sys
import urllib.request, urllib.error


class JupyterHubClient:
    def __init__(self, host, port, user, token, timeout=600):
        self.host    = host
        self.port    = int(port)
        self.user    = user
        self.token   = token
        self.timeout = timeout

    # ── Kernel management ────────────────────────────────────────────────────

    def list_kernels(self):
        url = f"http://{self.host}:{self.port}/user/{self.user}/api/kernels?token={self.token}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        try:
            result = json.loads(data)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []

    def start_server(self):
        """Start the JupyterHub single-user server if not running."""
        url = f"http://{self.host}:{self.port}/hub/api/users/{self.user}/server"
        req = urllib.request.Request(
            url, data=b"",
            headers={"Authorization": f"token {self.token}",
                     "Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status in (201, 202)
        except urllib.error.HTTPError as e:
            if e.code == 400:  # already running
                return True
            raise

    def new_kernel(self, kernel_name="python3"):
        url = f"http://{self.host}:{self.port}/user/{self.user}/api/kernels?token={self.token}"
        req = urllib.request.Request(
            url, data=json.dumps({"name": kernel_name}).encode(),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())["id"]
        except urllib.error.HTTPError as e:
            if e.code == 405:
                # Single-user server not running — start it and retry
                self.start_server()
                time.sleep(3)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read())["id"]
            raise

    def get_or_create_kernel(self, kernel_name="python3"):
        for k in self.list_kernels():
            if k["name"] == kernel_name:
                return k["id"]
        return self.new_kernel(kernel_name)

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def _handshake(self, sock, kernel_id):
        path = f"/user/{self.user}/api/kernels/{kernel_id}/channels?token={self.token}"
        key  = base64.b64encode(os.urandom(16)).decode()
        req  = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += sock.recv(4096)
        if b"101" not in resp:
            raise ConnectionError(f"WebSocket handshake failed: {resp[:200]}")

    def _ws_send(self, sock, data):
        if isinstance(data, str):
            data = data.encode()
        mask   = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        length = len(data)
        if length < 126:
            header = bytes([0x81, 0x80 | length]) + mask
        elif length < 65536:
            header = bytes([0x81, 0xFE]) + struct.pack(">H", length) + mask
        else:
            header = bytes([0x81, 0xFF]) + struct.pack(">Q", length) + mask
        sock.sendall(header + masked)

    def _ws_recv(self, sock):
        def recv_exact(n):
            buf = b""
            while len(buf) < n:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError("Socket closed")
                buf += chunk
            return buf

        b1, b2   = recv_exact(2)
        opcode   = b1 & 0x0F
        masked   = (b2 & 0x80) != 0
        length   = b2 & 0x7F
        if length == 126:
            length = struct.unpack(">H", recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", recv_exact(8))[0]
        mask_key = recv_exact(4) if masked else b""
        payload  = recv_exact(length)
        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        if opcode == 8:
            return None
        if opcode == 9:
            self._ws_send(sock, b"")
            return ""
        return payload.decode("utf-8", errors="replace")

    # ── Execute ───────────────────────────────────────────────────────────────

    def execute(self, code, kernel_id=None, stdout=None):
        """
        Execute code on the kernel. Streams output to stdout (default: sys.stdout).
        Returns the kernel execution status ('ok' or 'error').
        """
        if stdout is None:
            stdout = sys.stdout
        if kernel_id is None:
            kernel_id = self.get_or_create_kernel()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.settimeout(self.timeout)

        self._handshake(sock, kernel_id)

        msg_id = str(uuid.uuid4())
        execute_msg = {
            "header": {
                "msg_id": msg_id, "username": self.user,
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.3", "date": ""
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code, "silent": False,
                "store_history": False,
                "user_expressions": {}, "allow_stdin": False
            },
            "channel": "shell",
            "buffers": []
        }
        self._ws_send(sock, json.dumps(execute_msg))

        status   = "unknown"
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                frame = self._ws_recv(sock)
            except socket.timeout:
                continue
            if frame is None:
                break
            if not frame:
                continue
            try:
                msg = json.loads(frame)
            except json.JSONDecodeError:
                continue

            mt = msg.get("msg_type", "")
            if mt == "stream":
                stdout.write(msg["content"]["text"])
                stdout.flush()
            elif mt in ("execute_result", "display_data"):
                stdout.write(msg["content"]["data"].get("text/plain", "") + "\n")
                stdout.flush()
            elif mt == "error":
                sys.stderr.write("KERNEL ERROR: " + msg["content"]["evalue"] + "\n")
                for line in msg["content"]["traceback"]:
                    sys.stderr.write(line + "\n")
                status = "error"
                break
            elif mt == "execute_reply":
                status = msg["content"]["status"]
                break

        sock.close()
        return status


# ── Module-level convenience functions ────────────────────────────────────────

def _client_from_env():
    """Build a JupyterHubClient from environment variables or .env file."""
    import pathlib

    def load_env():
        for candidate in [pathlib.Path.cwd() / ".env", pathlib.Path.home() / ".env"]:
            if candidate.exists():
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip())
                break

    load_env()
    return JupyterHubClient(
        host    = os.getenv("JH_HOST",    "localhost"),
        port    = os.getenv("JH_PORT",    "8000"),
        user    = os.getenv("JH_USER",    ""),
        token   = os.getenv("JH_TOKEN",   ""),
        timeout = int(os.getenv("JH_TIMEOUT", "600")),
    )


def execute(code, kernel_id=None):
    return _client_from_env().execute(code, kernel_id=kernel_id)


def list_kernels():
    return _client_from_env().list_kernels()


def new_kernel(kernel_name="python3"):
    return _client_from_env().new_kernel(kernel_name)


def get_or_create_kernel(kernel_name="python3"):
    return _client_from_env().get_or_create_kernel(kernel_name)

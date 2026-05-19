"""Network helpers."""
from __future__ import annotations

import socket


def local_ip_toward(host: str, port: int = 80) -> str:
    """Return the local IP address the OS would use to reach `host`.

    We do this by opening a UDP socket (no packets actually sent) and asking what
    sockname it picked. Works without DNS for IP literals and across multiple NICs.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, port))
        return sock.getsockname()[0]
    finally:
        sock.close()

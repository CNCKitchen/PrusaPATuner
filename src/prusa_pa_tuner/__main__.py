"""`python -m prusa_pa_tuner` — start the server and open a browser."""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser

import uvicorn

from . import __version__
from .config import config_path, load_config


def main() -> int:
    parser = argparse.ArgumentParser(prog="prusa-pa-tuner", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_config()
    print(f"PrusaPATuner v{__version__}")
    print(f"  config: {config_path()}")
    print(f"  http:   http://{args.host}:{args.port}/")
    print(f"  udp:    listening on port {cfg.udp_port}")
    print()

    if not args.no_browser:
        def _open():
            time.sleep(0.8)
            try:
                webbrowser.open(f"http://{args.host}:{args.port}/")
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "prusa_pa_tuner.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

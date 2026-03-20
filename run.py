#!/usr/bin/env python3
"""Start the recaper web server."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="recaper web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(
        f"\n"
        f"  ╔══════════════════════════════════════╗\n"
        f"  ║          recaper web server           ║\n"
        f"  ╠══════════════════════════════════════╣\n"
        f"  ║  http://{args.host}:{args.port:<24}║\n"
        f"  ╚══════════════════════════════════════╝\n"
    )

    uvicorn.run(
        "recaper.web.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()

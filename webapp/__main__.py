"""Arranca la app web local:  uv run python -m webapp  [--port 8000] [--no-browser]

Todo corre en esta maquina; el servidor solo escucha en 127.0.0.1.
"""
import argparse
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Padel Pose en vivo: {url}  (Ctrl+C para cerrar)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("webapp.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

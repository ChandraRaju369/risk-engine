"""One command to get ready for the live demo:  python -m scripts.demo_prep

Deletes the old database, loads the news and social replays, builds the dashboard,
and warms up the models so nothing is slow while you are presenting.
"""
import subprocess
import sys

from engine.config import DB_PATH


def run(*args):
    print("\n>>", " ".join(args))
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Deleted old database.")
    run("engine.pipeline", "sample_replay.jsonl")
    run("engine.pipeline", "demo_social.jsonl")
    run("module_a.run")
    print("\nReady. For the demo:")
    print("  1. python -m uvicorn engine.api:app      -> open http://127.0.0.1:8000/docs")
    print("  2. open output\\dashboard.html")
    print("  3. open output\\backtest.html")


if __name__ == "__main__":
    main()

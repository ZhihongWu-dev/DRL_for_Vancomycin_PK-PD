"""Package-local TensorBoard launcher for IQL package.

Usage:
  python -m algorithms.iql.run_tensorboard --logdir runs --port 6006
"""
import os
import sys
import argparse
import subprocess
# ensure project root is on sys.path when running the script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs", help="TensorBoard logdir")
    p.add_argument("--port", type=int, default=6006, help="Port for TensorBoard")
    args = p.parse_args(argv)

    # Prefer programmatic API when available (more robust across environments)
    try:
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", args.logdir, "--port", str(args.port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}")
        # attempt to open the browser automatically; ignore failures (headless envs)
        try:
            import webbrowser
            webbrowser.open(url, new=2)
            print(f"Opened browser to {url}")
        except Exception:
            print("Could not open browser automatically; please open the URL above in a browser.")
    except Exception as e:
        # Fallback to attempting to run the tensorboard module as a subprocess
        cmd = [sys.executable, "-m", "tensorboard", "--logdir", args.logdir, "--port", str(args.port)]
        print(f"Could not start via TensorBoard API ({e}). Trying subprocess: {' '.join(cmd)}")
        try:
            subprocess.run(cmd)
        except FileNotFoundError:
            print("TensorBoard is not installed or not executable. Install with: pip install tensorboard")


if __name__ == "__main__":
    main()
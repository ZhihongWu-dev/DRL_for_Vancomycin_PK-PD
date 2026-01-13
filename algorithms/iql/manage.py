"""Utilities to manage IQL experiments: start TensorBoard with retries, launch training and monitor.

Usage examples:
  python -m algorithms.iql.manage start-tb --logdir algorithms/iql/runs --port 6006
  python -m algorithms.iql.manage run-and-monitor --config configs/iql_base.yaml --workdir algorithms/iql/runs/exp1
"""
import argparse
import os
import subprocess
import sys
import time
from urllib.request import urlopen


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        urlopen(f"http://{host}:{port}/", timeout=timeout)
        return True
    except Exception:
        return False


def start_tensorboard(logdir: str, start_port: int = 6006, max_tries: int = 5, host: str = "127.0.0.1"):
    """Start TensorBoard in background trying a sequence of ports. Returns (proc, url, port).
    stdout/stderr are redirected to `tb_log_port{port}.txt` in the logdir.
    """
    logdir = os.path.abspath(logdir)
    os.makedirs(logdir, exist_ok=True)

    for i in range(max_tries):
        port = start_port + i
        tb_log = os.path.join(logdir, f"tb_log_port{port}.txt")
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port), "--host", host]
        print("Starting TensorBoard:", " ".join(cmd))
        with open(tb_log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        # wait a short time and check
        for _ in range(10):
            time.sleep(0.5)
            if _is_port_open(host, port, timeout=0.5):
                url = f"http://{host}:{port}/"
                print(f"TensorBoard started at {url} (log: {tb_log})")
                # try opening browser, ignore failures
                try:
                    import webbrowser
                    webbrowser.open(url, new=2)
                except Exception:
                    pass
                return proc, url, port
        # if not open, terminate and try next port
        print(f"Port {port} not responding, killing process and trying next port")
        try:
            proc.kill()
        except Exception:
            pass
    raise RuntimeError("Failed to start TensorBoard on ports {start_port}..{start_port + max_tries - 1}")


def run_training_subprocess(config_path: str, workdir: str = None) -> subprocess.Popen:
    """Start training as a subprocess. Returns the Popen process.
    Expects that the training entrypoint is `algorithms/iql/train_iql.py` and accepts `--config`.
    """
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train_iql.py"), "--config", config_path]
    env = os.environ.copy()
    if workdir:
        env["IQL_WORKDIR"] = workdir
    print("Launching training:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc


def run_and_monitor(config: str, workdir: str = None, tb_port: int = 6006):
    """Run training and start tensorboard; training runs in foreground (blocking)."""
    # If config looks like an in-repo path, ensure it exists
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file not found: {config}")

    # Start TensorBoard (non-blocking)
    try:
        tb_proc, url, port = start_tensorboard(logdir=workdir or "algorithms/iql/runs", start_port=tb_port)
    except Exception as e:
        print("Warning: could not start TensorBoard:", e)
        tb_proc = None

    # Launch training
    train_proc = run_training_subprocess(config, workdir=workdir)
    print("Training process started (PID: %s)." % train_proc.pid)
    print("Training runs in the foreground. Press Ctrl+C to terminate.")

    try:
        train_proc.wait()
    except KeyboardInterrupt:
        print("Interrupted: terminating training process")
        try:
            train_proc.terminate()
        except Exception:
            pass

    if tb_proc is not None:
        print("TensorBoard left running in background (PID: %s)." % tb_proc.pid)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    p_tb = sub.add_parser("start-tb")
    p_tb.add_argument("--logdir", default="algorithms/iql/runs", help="TensorBoard logdir")
    p_tb.add_argument("--port", type=int, default=6006)
    p_tb.add_argument("--tries", type=int, default=5)

    p_run = sub.add_parser("run-and-monitor")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--workdir", default=None)
    p_run.add_argument("--tb-port", type=int, default=6006)

    args = p.parse_args(argv)
    if args.cmd == "start-tb":
        start_tensorboard(args.logdir, start_port=args.port, max_tries=args.tries)
    elif args.cmd == "run-and-monitor":
        run_and_monitor(args.config, workdir=args.workdir, tb_port=args.tb_port)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

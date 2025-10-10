#!/usr/bin/env python3
"""Start lightweight static hosts for model files based on config/files/hosting.yml

This script reads config/files/hosting.yml and for each entry starts a simple
HTTP server serving the configured `serve_dir` on the configured port.

It writes PID files to host/pids/<model>.pid so `stop_hosts.sh` can stop them.
"""
import yaml
import subprocess
import sys
from pathlib import Path
import argparse
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config' / 'files' / 'hosting.yml'
PIDS_DIR = ROOT / 'host' / 'pids'

PIDS_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description='Start model file hosts from hosting.yml')
parser.add_argument('--foreground', action='store_true', help='Do not daemonize; run in foreground')
parser.add_argument('--list', action='store_true', help='List planned hosts and exit (dry-run)')
args = parser.parse_args()

# Try to import hosting config via the project's config loader package (non-destructive)
cfg = None
try:
    # Ensure the repo root is on PYTHONPATH if called from repo root
    # Importing `config.loader` will use config/loader/__init__.py which exposes `hosting` if present
    from config.loader import hosting as hosting_cfg  # type: ignore
    if isinstance(hosting_cfg, dict) and hosting_cfg:
        cfg = dict(hosting_cfg)
except Exception:
    # If import fails (e.g., PYTHONPATH not set), we'll fall back to reading the YAML file below
    cfg = None

if cfg is None:
    if not CONFIG_PATH.is_file():
        print('hosting config not found at', CONFIG_PATH, file=sys.stderr)
        sys.exit(2)

    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f) or {}

# If there is a _defaults entry, use it
defaults = cfg.get('_defaults', {})

started = []
for name, info in cfg.items():
    if name == '_defaults':
        continue
    host = info.get('host', defaults.get('host', '0.0.0.0'))
    port = int(info.get('port', defaults.get('base_port', 9000)))
    serve_dir = info.get('serve_dir', defaults.get('serve_root', 'host/models') + '/' + name)
    serve_path = Path(serve_dir)
    # If serve_dir is relative, interpret relative to repo root
    if not serve_path.is_absolute():
        serve_path = (ROOT / serve_dir).resolve()
    if not serve_path.is_dir():
        print(f'Creating serve dir for {name}:', serve_path)
        serve_path.mkdir(parents=True, exist_ok=True)
        # create a README file so directory is not empty
        (serve_path / 'README.txt').write_text(f'This folder hosts files for model {name}\n')

    print(f'Planned host: {name} -> http://{host}:{port} serving {serve_path}')
    if args.list:
        continue

    # start a simple http server
    cmd = [sys.executable, '-m', 'http.server', str(port), '--bind', host]
    # Use --directory if Python supports it
    try:
        proc = subprocess.Popen(cmd + ['--directory', str(serve_path)])
    except Exception:
        # Fallback: change cwd
        proc = subprocess.Popen(cmd, cwd=str(serve_path))

    pid = proc.pid
    pidfile = PIDS_DIR / f'{name}.pid'
    pidfile.write_text(str(pid))
    print(f'Started {name} at http://{host}:{port} (pid={pid}) serving {serve_path}')
    started.append((name, host, port, pid))
    if args.foreground:
        try:
            proc.wait()
        except KeyboardInterrupt:
            print('Interrupted, exiting')
            break

if not args.foreground and not args.list:
    print('All hosts started. PID files are in', PIDS_DIR)
    time.sleep(0.1)

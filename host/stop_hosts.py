#!/usr/bin/env python3
"""Stop hosts started by start_hosts.py by reading PID files in host/pids
"""
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIDS_DIR = ROOT / 'host' / 'pids'

if not PIDS_DIR.exists():
    print('No PID directory found at', PIDS_DIR)
    raise SystemExit(0)

for pidfile in PIDS_DIR.glob('*.pid'):
    try:
        pid = int(pidfile.read_text().strip())
    except Exception as e:
        print('Failed to read pid from', pidfile, e)
        continue
    try:
        print('Killing pid', pid)
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print('Process not found, removing pidfile', pidfile)
    except PermissionError:
        print('Permission denied killing pid', pid)
    finally:
        try:
            pidfile.unlink()
        except Exception:
            pass

print('Done')


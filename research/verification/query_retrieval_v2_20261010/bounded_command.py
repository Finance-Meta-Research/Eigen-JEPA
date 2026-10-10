"""Local command recorder for this finite engineering pass, not a study runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    record = args.cwd / args.record
    record.mkdir(parents=True, exist_ok=False)
    def sources():
        return {name: hashlib.sha256((args.cwd / name).read_bytes()).hexdigest()
                for name in args.source}
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    receipt = {
        'command': command, 'cwd': str(args.cwd), 'timeout_seconds': args.timeout,
        'python': platform.python_version(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': {name: env.get(name) for name in
                        ('PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')},
        'source_sha256_before': sources(), 'scientific_outcomes': False,
    }
    (record / 'started.json').write_text(json.dumps(receipt, indent=2)+'\n')
    start = time.monotonic()
    with (record / 'stdout.log').open('wb') as out, (record / 'stderr.log').open('wb') as err:
        process = subprocess.Popen(command, cwd=args.cwd, env=env, stdout=out, stderr=err,
                                   start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    receipt.update(wall_seconds=time.monotonic()-start, exit_code=process.returncode,
                   timed_out=timed_out, source_sha256_after=sources(),
                   child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
    receipt['source_unchanged'] = receipt['source_sha256_before'] == receipt['source_sha256_after']
    receipt['output_sha256'] = {name: hashlib.sha256((record/name).read_bytes()).hexdigest()
                               for name in ('stdout.log', 'stderr.log')}
    (record / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    print((record / 'stdout.log').read_text(errors='replace')[-16000:])
    print((record / 'stderr.log').read_text(errors='replace')[-8000:])
    return process.returncode if process.returncode >= 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

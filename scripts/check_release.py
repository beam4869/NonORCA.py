"""Run the release's data checks and existing targeted Python tests."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ['scripts/reproduce_available.py', '--skip-plots'],
    ['scripts/verify_local_sources.py'],
    ['-m', 'pytest', '-q',
     'tests/core/test_constants.py',
     'tests/core/test_nonlinear_benchmarks.py',
     'tests/core/test_constrained_benchmarks.py',
     'tests/core/test_experiment_harness.py',
     'tests/dtlz9/test_dtlz9.py'],
]
if __name__ == '__main__':
    for args in COMMANDS:
        print('Running:', sys.executable, *args, flush=True)
        result = subprocess.run([sys.executable, *args], cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
    print('Release checks passed. Julia NLP solves are outside these checks.')

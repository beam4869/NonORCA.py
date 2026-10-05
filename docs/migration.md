# Repository reorganization — 2026-10-05

The repository now uses one installable Python library plus separate paper
workflows. The previous preparation was `ORCA_GitHub_Preparation.zip`.

| Previous location | Current location |
|---|---|
| `ORCA_python/` implementation | `src/orca/` |
| `ORCA_python/src/` legacy linear translation | `src/orca/legacy/` |
| `ORCA_python/test/` | `tests/core/` |
| Root `run_*.py` drivers | `experiments/dtlz/` |
| `dtlz9_experiments/` | `experiments/dtlz9/` |
| `dtlz9_experiments/test_dtlz9.py` | `tests/dtlz9/test_dtlz9.py` |
| `ccus_orca_experiments/` | `experiments/ccus/` |
| `Nonlinear_demonstration_code/` | `examples/ellipse/` |
| Embedded Poetry project metadata | `archive/package_metadata/` |
| Embedded Poetry lock | `environments/historical/poetry.lock` |

## What changed

- Public imports use `orca` instead of `ORCA_python`. Install with
  `python -m pip install -e .` before running scripts. No duplicate package or
  hidden compatibility copy is maintained.
- The new root `pyproject.toml` defines a setuptools package with optional
  Leiden, Pyomo, experiment and test dependencies. The base library requires
  only NumPy and SciPy; the original Poetry environment remains historical.
- Repository-relative paths were adjusted for the moved drivers. Numerical
  routines, configuration defaults and archived data were not revised.
- Default pytest discovery selects the existing 41-test manuscript subset.
  The historical NH3 tests and CCUS export-contract tests are retained but not
  newly claimed as passing in the default suite.
- English and Chinese READMEs now describe software usage rather than the
  recovery process. API, reproduction and figure/table documentation were updated.

## Provenance

`source_manifest.json` retains the original uploaded source paths and SHA-256
hashes, adds the prior preparation paths/hashes, and records current paths and
hashes. `reorganization_manifest.json` accounts for every file in the previous
724-file preparation, including documentation and generated evidence not listed
in the original source manifest. The old manifest is retained under
`archive/package_metadata/preparation_source_manifest.json`.

Historical reports preserve their original paths, dates and numerical claims;
they are evidence of those earlier runs. Use current documentation for commands.
The migration does not resolve the numerical, missing-data or attribution
limitations listed in `known_issues.md`.

No Git remote, commit or publication was created by this reorganization.

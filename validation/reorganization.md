# Layout migration validation — 2026-10-05

Validation ran on Linux with Python 3.12.14. It verifies the new layout and
preservation of stored manuscript results; it does not establish exact
historical optimization reproduction.

| Check | Result |
|---|---|
| Editable installation from root `pyproject.toml` | Passed |
| `python examples/quickstart.py` | Passed; 11 selected points, groups `[1, 1, 2]` |
| Wheel build and installation into a separate environment | Passed |
| Quickstart using installed wheel from outside the repository | Passed, without the experiment or Leiden extras |
| Default `python -m pytest -q` | **41 passed** in 17.82 s |
| Source integrity and stored-table audit | **727 checks passed** |
| Ellipse, Table 1/2 and CCUS computational audits | **30 checks passed** |
| Four relocated DTLZ5/6 driver imports | Passed |
| DTLZ9 command-line entry point | `--help` passed |
| Archived-data overview and CCUS plotting entry points | Passed |
| Core abstract-syntax comparison after import renaming | **35 modules unchanged** apart from import paths |
| Numerical-data and Julia source byte comparison | **470 files unchanged** |
| Current Python source syntax | Passed |
| Current documentation links | Checked after documentation completion |
| Git ignore coverage | No source-manifest input is excluded by `.gitignore` |

The quickstart is an API example, not a paper run. The wheel contains the
library and distribution metadata, not experiment data, test fixtures or
archived results. Records are in [reorganization/](reorganization/).

The default manuscript tests intentionally exclude historical NH3 tests and
CCUS tests tied to omitted image exports. Fresh full DTLZ optimization sweeps
and Julia solves were not run for this layout-only change. Existing numerical
drift, missing Fig. 4 input and Table 2 SD limitations remain in
[known issues](../docs/known_issues.md).

The older files elsewhere in `validation/` record the previous preparation;
their timestamps and result claims should not be read as new executions.

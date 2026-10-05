# Upload to an empty GitHub repository

Extract the delivered ZIP. The directory containing `README.md`, `pyproject.toml`
and `src/` is the repository root; do not add an extra enclosing directory.
Use that directory in Terminal and replace the remote URL below with the URL
of your empty repository:

```bash
cd /path/to/ORCA
git init
git add .
git status --short
git commit -m "Add ORCA package and manuscript experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

These commands assume the target repository is empty and this directory has
not already been initialized. If using an existing checkout, copy the files
into that checkout and use its existing branch and remote instead.

The ignore file excludes virtual environments, caches and fresh outputs.
CSV inputs, Julia project/manifest files, source code, tests and documentation
are included. The largest original data file is below GitHub's 100 MiB
regular-file limit; no Git LFS conversion was made.

Before a manuscript release, run `python scripts/check_release.py`. The GitHub
Actions workflow runs the quickstart and checks after a push; local validation
does not mean that a remote workflow has already run.

When a paper snapshot is ready, create a tag/release for that exact commit and
refer readers to it. Keep the stated reproduction limits in the README. This
preparation itself has not created a Git commit or pushed to a remote.

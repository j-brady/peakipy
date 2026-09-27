# Release process

How peakipy is versioned, tested and published. Written against the `v2.2.1`
release, which was the first release published through the current
uv/Trusted-Publishing pipeline.

## How publishing works

There is no manual upload step and no API token to manage. `.github/workflows/release.yml`
triggers on any tag matching `v[0-9]+.[0-9]+.[0-9]+` and then:

1. asserts that the tag matches the version in `pyproject.toml`
   (`test "$(uv version --short)" = "${GITHUB_REF_NAME#v}"`),
2. runs `uv build`,
3. runs `uv publish` against PyPI using **Trusted Publishing** (OIDC), scoped to
   the `pypi` environment.

Because step 1 is an assertion, a tag that disagrees with `pyproject.toml` fails
the run rather than publishing something mislabelled. Note also that
`uv publish` is not a step you can usefully run by hand — there is no OIDC token
outside CI.

## Checklist

1. **Work on a branch and open a PR.** `master` has branch protection requiring
   **one approving review**. GitHub will not let a PR author approve their own
   PR, so if you opened it you need either another reviewer or to merge it
   yourself as an admin.

2. **Bump the version.** The version lives in `pyproject.toml` and nowhere else.

   ```bash
   uv version --bump patch   # or minor / major
   ```

   This updates `pyproject.toml` and re-locks as needed.

3. **Add a `CHANGELOG.md` entry.** Keep a Changelog style: a
   `## [x.y.z] - YYYY-MM-DD` heading with `### Added` / `### Changed` /
   `### Fixed` subsections, newest first.

4. **Verify locally.** This is the real gate — do not skip it because CI is
   green on someone else's unrelated change.

   ```bash
   uv lock --check        # uv.lock is in sync with pyproject.toml
   uv run make coverage   # the same test selection CI runs
   uv run mkdocs build    # see the docs note below
   ```

5. **Merge, then tag and push the tag.**

   ```bash
   git tag -a v2.2.1 -m "v2.2.1"
   git push origin v2.2.1
   ```

6. **Confirm the publish.** Watch the "Publish release to PyPI" run to success.
   PyPI's JSON API can lag a minute or two behind a successful workflow, so
   re-check once before assuming the upload failed.

7. **Create the GitHub release** so the tag has release notes:

   ```bash
   gh release create v2.2.1 --title "v2.2.1" --notes "..."
   ```

## Dependency updates

CI installs with `uv sync --locked`, so **`uv.lock` must stay consistent with
`pyproject.toml`** or the build fails. `uv lock --check` verifies this locally.

For anything that is not a routine version bump, prefer targeted upgrades over a
blanket one. A bare `uv lock --upgrade` pulls in unrelated major bumps across
the whole graph — as of the 2.2.1 release it wanted typer 0.17 → 0.27,
rich 14 → 15, scipy → 1.18 and statsmodels 0.14 → 0.15, none of which were
wanted. To move only what you need:

```bash
uv lock --upgrade -P <package> -P <package>   # add --dry-run to preview
```

The two cases are different:

- **Transitive dependency** (pillow, urllib3, tornado, asteval): lock-only is
  enough. Nothing in `pyproject.toml` needs to change, because a fresh install
  already resolves to the newest compatible version.
- **Direct dependency whose declared floor sits inside an advisory's vulnerable
  range** (this was bokeh, black, mkdocs-material, pytest): raise the floor in
  `pyproject.toml`, otherwise a new install can still resolve to a vulnerable
  version.

Use *first patched* versions as floors rather than *newest* versions. The
floor is a compatibility promise; the lock can sit further ahead.

### Dependabot

`.github/dependabot.yml` watches the `uv` ecosystem weekly and groups all
updates into a single pull request. Two consequences worth knowing:

- Grouped PRs can bundle several major bumps at once. A green CI run is not
  sufficient reason to merge one — check the diff for majors (`pandas` 3.0 and
  `plotly` 7 both arrived bundled in the first such PR) and treat those as
  deliberate work, not maintenance.
- The config does not exclude major updates from the group. If you would
  rather review majors one at a time, add
  `update-types: ["minor", "patch"]` to the group.

Dependabot only edits `uv.lock` for indirect updates; it will not raise a floor
in `pyproject.toml` for you.

## What CI actually runs

`ci.yml` tests Python 3.11, 3.12 and 3.13 with `uv sync --locked --no-editable`
— `--no-editable` so the tests run against the installed build rather than the
working tree. Actions are SHA-pinned; keep that convention when editing
workflows.

`make coverage` runs an explicit list of test files. `test/test_data.py` and
`test/test_edit.py` are **not** in that list, so they do not run in CI. If you
touch the code they cover, run them by hand.

## Known rough edges

- **`docs.yml` is outside the lock.** It runs on every push to `master` and does
  a bare `pip install mkdocs-material`, so it picks up the newest docs stack
  whether or not the lock does, and it still uses the deprecated
  `actions/checkout@v2` / `actions/setup-python@v2`. This is why step 4 tells
  you to run `uv run mkdocs build` — a docs-tooling major can break the site
  independently of anything the tests cover.
- **Formatting is not enforced.** Nothing in CI runs `black`, and the repo has
  pre-existing drift. `.pre-commit-config.yaml` pins its own `black` in a
  separate venv from the dev group, so the two can disagree; bump that `rev`
  when you move the dev group floor. Avoid reformatting inside a dependency
  pull request — it buries the actual change.
- **The sdist is large** (~28 MB) because the `test/` fixture directories are
  included. Pre-existing, and deliberately left alone.

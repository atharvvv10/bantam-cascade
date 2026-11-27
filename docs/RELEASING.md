# Releasing

Bantam Cascade publishes to PyPI with **Trusted Publishing**. No PyPI API token is stored in the repository, CI, or local `.env`.

## One-time PyPI setup

1. Create and verify a PyPI account at <https://pypi.org/account/register/>.
2. Enable two-factor authentication.
3. For an existing project such as `bantam-cascade`, go to the project's
   **Settings → Publishing** and add an **active publisher**; a pending
   publisher is only for a project that does not exist yet.
4. Enter:
   - PyPI project name: `bantam-cascade`
   - Owner: `atharvvv10`
   - Repository name: `bantam-cascade`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`

The pending publisher creates the PyPI project the first time the workflow
publishes. TestPyPI and PyPI have separate accounts and separate publisher
registries, so both must be configured explicitly.

## TestPyPI first

1. Add a pending publisher on <https://test.pypi.org/manage/account/publishing/>
   with owner `atharvvv10`, repository `bantam-cascade`, workflow
   `publish-testpypi.yml`, and environment `testpypi`.
2. Run the manual `Publish to TestPyPI` workflow.
3. Install from TestPyPI in a clean venv and run `bantam-cascade --version`,
   `bantam-cascade plan`, `bantam-cascade validity --trials 2 --examples 40`, and
   `bantam-cascade status-demo`. `reproduce` also verifies the frozen certificate
   and therefore needs the git-ignored run artifacts from a full source checkout.

No token is needed; both workflows use short-lived OIDC credentials. TestPyPI
accounts are separate from PyPI accounts. On 2026-11-27 the first dispatch (run
`45947996121`) failed with `invalid-publisher` because the publisher had been
configured on PyPI rather than TestPyPI. After creating the TestPyPI account and
adding the pending publisher there, run `45950924942` succeeded and published
`0.1.1`. A clean venv install from TestPyPI passed `--version`, `plan`, `validity`,
and `status-demo`. The OIDC subject is
`repo:atharvvv10@160210787/bantam-cascade@1469926336:environment:testpypi`; the
exact claim values are rendered by the workflow for debugging and must match the
TestPyPI publisher entry.

## Publishing

The `.github/workflows/publish.yml` workflow runs on a published GitHub Release or manual dispatch. It builds the sdist and wheel, checks them with Twine, and publishes through short-lived OIDC credentials from GitHub Actions.

On 2026-11-27 the release-triggered run `45951318395` failed with
`invalid-publisher` because production PyPI had no matching active publisher.
After configuring `publish.yml` with environment `pypi` on PyPI, the manual run
`45951719762` succeeded and published `0.1.1`. The OIDC subject was
`repo:atharvvv10@160210787/bantam-cascade@1469926336:environment:pypi`. A clean
venv install from PyPI passed `--version`, `plan`, `validity`, and `status-demo`.
The first install attempt saw only `0.1.0` because the simple index had not yet
caught up with the JSON API; a retry succeeded.

The `v0.1.1` GitHub Release assets are the **exact files downloaded from PyPI**,
not a second local build. Their SHA256 values are:

- `bantam_cascade-0.1.1-py3-none-any.whl`: `986fab79a764c37a98fbdf26687dad71199d0224bd95fc51f003588e31ba0a42`
- `bantam_cascade-0.1.1.tar.gz`: `24d236efc39bc04bc6cf5519c5f711eb43dbd762c5c2467ec3e47d5435858c60`

The PyPI long description is the snapshot built from tag `v0.1.1`; later
main-branch documentation edits do not alter the already-published metadata.

Do not create or commit a long-lived PyPI API token unless Trusted Publishing cannot be used. If an account-wide token is ever used for a bootstrap upload, revoke it immediately afterward and replace it with the trusted publisher above.

## Repository metadata

The README now carries the CI and PyPI badges and a real results table. On
2026-11-27 the repository description and topics were set through the GitHub REST
API using the existing Git credential; no token was printed or stored. The
equivalent maintainer command is kept here for reference:

```bash
gh repo edit atharvvv10/bantam-cascade \
  --description "Certify cheap small-model + LLM fallback cascades for text classification" \
  --add-topic text-classification --add-topic calibration --add-topic cascade \
  --add-topic certification --add-topic statistics --add-topic cpu
```

No token or key is stored in the repository; run this where `gh auth status`
already succeeds.

# Releasing

Bantam Cascade publishes to PyPI with **Trusted Publishing**. No PyPI API token is stored in the repository, CI, or local `.env`.

## One-time PyPI setup

1. Create and verify a PyPI account at <https://pypi.org/account/register/>.
2. Enable two-factor authentication.
3. Go to account settings → **Publishing** → **Add pending publisher** → **GitHub Actions**.
4. Enter:
   - PyPI project name: `bantam-cascade`
   - Owner: `atharvvv10`
   - Repository name: `bantam-cascade`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`

The pending publisher creates the PyPI project the first time the workflow publishes.

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

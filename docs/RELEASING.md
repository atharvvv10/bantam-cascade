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
3. Install from TestPyPI in a clean venv and run `bantam-cascade --version` plus
   `bantam-cascade reproduce --trials 2 --examples 40`.

No token is needed; both workflows use short-lived OIDC credentials. On
2026-11-27 the first TestPyPI dispatch (run `45947996121`) built successfully but
failed with `invalid-publisher`: TestPyPI has no matching pending publisher yet.
Configure it with owner `atharvvv10`, repository `bantam-cascade`, workflow
`publish-testpypi.yml`, environment `testpypi`, then re-run the manual workflow.
The OIDC subject shown by GitHub was
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

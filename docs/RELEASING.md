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

## Publishing

The `.github/workflows/publish.yml` workflow runs on a published GitHub Release or manual dispatch. It builds the sdist and wheel, checks them with Twine, and publishes through short-lived OIDC credentials from GitHub Actions.

Do not create or commit a long-lived PyPI API token unless Trusted Publishing cannot be used. If an account-wide token is ever used for a bootstrap upload, revoke it immediately afterward and replace it with the trusted publisher above.

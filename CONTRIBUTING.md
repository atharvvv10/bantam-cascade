# Contributing to Bantam Cascade

Thanks for helping. Please keep the scope tight: this project is a focused certification tool, not a general LLM platform.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
mypy
pytest
```

CI runs the same checks on Python 3.11 and 3.12. It must never call a real API or require `GROQ_API_KEY`: teacher-client tests use mocked HTTP only.

## Ground rules

- Never hardcode benchmark numbers, prices, model IDs, or API keys.
- Accuracy is always measured against gold human labels, never teacher agreement.
- Every documented number must trace to a results file from a real run.
- Use `bantam_cascade` as the Python import name; never `bantam`.

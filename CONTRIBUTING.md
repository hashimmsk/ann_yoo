# Contributing to Dr Yoo Research

Thanks for your interest in improving the datanuri / ADJANN research stack! This document explains how to set up your environment, run checks, and submit changes.

## Development Workflow
1. Fork the repository and create a feature branch off `main`.
2. Install dependencies inside a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Run or update the application components as needed:
   - Backend API: `uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload`
   - Full stack: `python backend/start_app.py`
   - Report generation: `python generate_report.py`
4. Add or update tests before opening a pull request. (Unit tests live under `tests/` once added.)
5. Format and lint your code. We recommend `black` for formatting and `ruff` or `flake8` for linting. (Add a `pyproject.toml` entry if you introduce new tooling.)

## Pull Request Checklist
- Include context in the PR description: what changes you made and why.
- Update documentation, including the README or API reference, when behaviour changes.
- Add entries to `CHANGELOG.md` under the “Unreleased” section.
- Ensure CI checks pass (if configured).

## Reporting Issues
Open a GitHub issue with:
- Summary of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Environment details (OS, Python version, package versions)

Security vulnerabilities should be reported privately – see `SECURITY.md`.

## Review Guidelines
- Follow existing code style.
- Keep functions and modules focused; prefer small, testable units.
- Discuss large or breaking changes in an issue before implementation.

We appreciate your contributions to making this research tool better. Thank you!


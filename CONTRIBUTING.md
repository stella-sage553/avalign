# Contributing

Thanks for your interest in `avalign`. It's a small project, so the process is
light.

## Development setup

```bash
git clone https://github.com/stella-sage553/avalign
cd avalign
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # add ,torch for the torch tests
pre-commit install
```

## Before opening a PR

```bash
ruff check .
ruff format .
mypy avalign
pytest
```

- Keep the NumPy core free of heavy dependencies; anything needing `torch`
  belongs in `avalign.torchext` behind the import guard.
- Add or update a test alongside any behaviour change.
- Public functions get type hints and a short docstring.
- Commit messages are free-form; small, focused commits are appreciated.

## Reporting bugs

Open an issue with a minimal reproduction. The bug-report template lists the
details that help.

# Contributing to llmSHAP

Contributions of all sizes are welcome: bug reports, docs fixes, examples, new heuristics/value functions, tests, and performance improvements.

## Before you start

- Please search existing issues (and open a new one if needed) so work doesn’t get duplicated.
- For larger changes (new attribution method, new public API, refactors), it’s best to open an issue first describing the approach and trade-offs.

## Development setup

Clone your fork and install the full dependency setup in editable mode. That is, using:

```bash
pip install -e ".[all]"
```

The repo uses a `src/` layout (library code in `src/llmSHAP`) and tests in `tests/`.

## Running tests

```bash
pytest -vv
```

If you add new functionality, please add or update tests. 
Prefer small, deterministic unit tests. 
If a test depends on an external model/provider, keep it behind an explicit opt-in (e.g., a marker) and document the required environment variables.

## Local LLM/provider credentials

Some examples or optional tests may require API keys (e.g., OpenAI). Never commit secrets.

Recommended pattern:
- Export keys in your shell (or use a local `.env` that is ignored by git)
- Keep provider calls out of unit tests

## Building docs (optional)

Documentation lives in `docs/`. 
If you change public APIs, please update the docs and/or tutorial accordingly.

A common workflow is:

```bash
# from repo root
cd docs
make html
```

If your change affects user-facing behavior, add a short snippet to the docs/tutorial showing how to use it.

## Coding guidelines

- Keep changes focused and easy to review.
- Maintain backwards compatibility unless there’s a clear reason to break it (and then call it out in the PR).
- Prefer readable, well-factored code over cleverness.
- Public functions/classes should have docstrings that explain intent and usage.
- Keep type hints up to date for new/changed APIs.


## Adding new features

If you’re adding a new component (e.g., a new value function, codec, heuristic, or LLM interface), please aim for:

- A small, composable class/module with a clear interface
- At least one unit test (and a doc/example if it’s user-facing)
- Sensible defaults and clear error messages
- No hard dependency on a specific provider unless it’s isolated behind an optional extra

## Pull request checklist

Before opening a PR, please make sure:

- [ ] Tests pass locally (`pytest`)
- [ ] Added/updated tests for new behavior
- [ ] Docs/examples updated (if user-facing)
- [ ] No secrets or credentials are included
- [ ] PR description explains: what changed, why, and how to test it

## Reporting bugs

When opening an issue, please include:

- What you expected to happen vs what happened
- Minimal reproduction code (smallest prompt/data that triggers it)
- Your environment (OS, Python version, llmSHAP version)
- Any relevant logs/tracebacks
- Whether a specific provider/model was used

## License

By contributing, you agree that your contributions will be licensed under the project’s MIT License.

# Project management

## Structure / Architecture

- Main code will be under `src/{package_name}`. If it can be reused and imported, it must be here
- Project must be modular, nesting modules is okay and welcome. Bundle stuff together under the same submodule whenever it's possible. Only cli.py may contain main() and runnable entrypoints/subcommands. All other modules must expose descriptive, single-purpose functions/classes and must not contain argparse or CLI parsing.
- Auxiliary scripts go under `scripts`
- Don't use relative imports

## Tooling

### uv
- Always use [uv](https://docs.astral.sh/uv/) for project management through a pyproject.toml.
- Use uv run to run scripts instead of `.venv/bin/python`
- ALWAYS use `pre-commit run --all-files` after you edit python files

## CLIs
- Use [jsonargparse](https://jsonargparse.readthedocs.io/en/v4.50.0/) `auto_cli` to build your CLI
- CLI must be unique for the whole project, with subcommands for every entry
- Add CLI as a [project.scripts] entry

## Testing

- Use pytest

# Code style

- Be abstraction oriented. Exploit object composition when possible, rather than inheritance
- Use ruff pre-commits

## Docstrings

- Use Google python style without type hints

# Contributing

Bug reports, fixes and new task pairs are welcome.

## Reporting

- **Security issues:** report them privately, as described in [SECURITY.md](SECURITY.md). Do not open
  a public issue.
- **Bugs and questions:** open a GitHub issue. Include the task id, the command you ran and the
  relevant part of its output (`scores.json`, the scorer's log). Remove any API keys or proxy URLs first.

## Development setup

```bash
uv sync --extra dev --extra data --extra modal --extra analysis
uv run pytest -q          # unit tests, no GPU or API
uv run ruff check <files you changed>
```

The tests run without a GPU, network access or model weights. Do not add tests that need any of them.

## Pull requests

- Keep a pull request to one change, and add or update a test for it under `tests/`.
- To add a task pair, follow [docs/ADDING_A_TASK.md](docs/ADDING_A_TASK.md). `tests/test_spec_hygiene.py`
  checks that the pieces agree.
- A change to an oracle, a scorer or a prompt changes what the paper's numbers mean. Say in the
  description whether reported results move, and how you checked.
- Never commit keys, `.env` files, model weights, agent transcripts or the held-out batteries under
  `data/held_out/`. A secret scan runs on every pull request.
- Bumping vLLM, torch or setuptools: read the "Dependency pins" section of [SECURITY.md](SECURITY.md)
  first.

## Dual-use content

Some prompts in this repository tell AI agents to sabotage the models they train. Contributions that
add attack framings or monitor-evasion tactics must keep them inside the sandboxed evaluation setting
described in the README; see "A note on the contents" there.

## Code of conduct

Everyone taking part in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the MIT License (see
[LICENSE](LICENSE)).

# Release

Meeting-ASR publishes to PyPI from GitHub Releases through PyPI Trusted Publishing.
Do not store PyPI API tokens in GitHub Secrets.

## One-time PyPI setup

Create or claim the PyPI project `meeting-asr`, then add a GitHub Actions trusted publisher.
If the project does not exist yet, create a pending trusted publisher with the same values:

- Owner: `crhan`
- Repository: `meeting-asr`
- Workflow file: `publish.yml`
- Environment: `pypi`

The GitHub environment is intentional. Configure `pypi` in GitHub repository settings and require a reviewer before deployment if the repository is shared.

## Release checklist

1. Confirm the working tree is clean.
2. Update `version` in `pyproject.toml`.
3. Run `uv run pytest`.
4. Run `uv build`.
5. Create and publish a GitHub Release for tag `vX.Y.Z`.
6. Wait for the `Publish to PyPI` workflow.
7. Verify install from PyPI:

```bash
uv tool install meeting-asr --python 3.14 --reinstall --refresh
meeting-asr --version
```

## Why Trusted Publishing

Trusted Publishing lets GitHub Actions request short-lived PyPI credentials through OIDC. The workflow does not need a long-lived PyPI token, so leaked repository secrets cannot become a PyPI publish credential.

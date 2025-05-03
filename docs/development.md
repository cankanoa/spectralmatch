# Development Guide

This project provides this [Makefile](https://github.com/spectralmatch/spectralmatch/blob/main/Makefile) to streamline development tasks. Makefiles allow you to automate and organize common tasks, in this case to help you serve and deploy documentation, manage version tags, format and lint code, and run tests.

> **Installation instructions are on their own [page](installation.md)**

## Docs

### Serve docs locally
Runs a local dev server at http://localhost:8000.
```bash
make docs-serve
```

### Build static site
Generates the static site into the site/ folder.

```bash
make docs-build
```

### Deploy to GitHub Pages
Deploys built site using mkdocs gh-deploy.
```bash
make docs-deploy
```

## Versioning
Uses git tag to create annotated version tags and push them.
```bash
make tag version=1.2.3
```

## Code Formatting
Uses black for code formatting and ruff for linting.

### Format code
Automatically formats all Python files with black.
```bash
make format
```

### Check formatting
Checks that all code is formatted (non-zero exit code if not).
```bash
make check-format
```

### Lint code
Runs ruff to catch style and quality issues.
```bash
make lint
```

## Testing
Uses [pytest](https://docs.pytest.org/) to run the test suite.
```bash
make test
```
# Contributing Guide

This project includes a [Makefile](https://github.com/spectralmatch/spectralmatch/blob/main/Makefile) to streamline development tasks. Makefiles allow you to automate and organize common tasks, in this case to help serve and deploy documentation, manage version tags, format and lint code, and run tests.

> **Installation instructions are on their own [page](installation.md)**

---

## Collaboration Instructions

We welcome all contributions the project! Please be respectful and work towards improving the library. To get started:

1. [Create an issue](https://github.com/spectralmatch/spectralmatch/issues/new) describing the feature or bug or just to ask a question. Provide relevant context, desired timeline, any assistance needed, who will be responsible for the work, anticipated results, and any other details.

2. [Fork the repository](https://github.com/spectralmatch/spectralmatch/fork) and create a new feature branch.

3. Make your changes and add any necessary tests.

4. Open a Pull Request against the main repository.

---

## Design Guidelines and Philosophy
- Keep code concise and simple
- Adapt code for large datasets with windows, multiprocessing, progressive computations, etc
- Keep code modular and have descriptive names
- Create docstrings (Google style), tests, and update the docs for new functionality
- Use similar naming convention and input parameters as other functions
- Use PEP 8 code formatting 

## File Cleanup
Temporary generated files can be deleted once they are no longer needed via this command:
```bash
make clean
```

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
---

## Versioning
Uses git tag to create annotated version tags and push them. This also syncs to Pypi. New versions will be released when the maintainer determines sufficient new functionality has been added.
```bash
make tag version=1.2.3
```

---

## Code Formatting
This project uses [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) for code formatting and ruff for linting.

### Set Up Pre-commit Hooks (Recommended)
To maintain code consistency use this hook to check and correct code formatting automatically:

```bash
pre-commit install
pre-commit run --all-files
```

### Manual Formatting

**Format code:** Automatically formats all Python files with black.

```bash
make format
```

**Check formatting:** Checks that all code is formatted (non-zero exit code if not).
```bash
make check-format
```

**Lint code:** Runs ruff to catch style and quality issues.
```bash
make lint
```

---

## Testing
[pytest](https://docs.pytest.org/) is used for testing. Tests will automatically be run when merging into main but they can also be run locally via:
```bash
make test
```

To test a individual folder or file:
```bash
make test-file path=path/to/folder_or_file
```
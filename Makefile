# Install
install:
	pip install .

install-dev:
	pip install -e '.[dev]'

install-docs:
	pip install -e '.[docs]'


# Docs
docs-serve:
	mkdir -p docs/images
	cp -r images/* docs/images/
	mkdocs serve -a localhost:8001

docs-build:
	mkdir -p docs/images
	cp -r images/* docs/images/
	mkdocs build

docs-deploy:
	mkdir -p docs/images
	cp -r images/* docs/images/
	mkdocs gh-deploy


# Versions
tag:
	@if [ -z "$(version)" ]; then \
		echo "Usage: make tag version=1.2.3"; \
		exit 1; \
	fi
	git tag -a v$(version) -m "Version $(version)"
	git push origin v$(version)


# Code formatting
format:
	black .

check-format:
	black --check .

lint:
	ruff check .


# Testing
test:
	pytest


# Cleanup
clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache


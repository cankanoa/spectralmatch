MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Install
install:
	pip install $(MAKEFILE_DIR).

install-dev:
	pip install -e '$(MAKEFILE_DIR).[dev]'

install-docs:
	pip install -e '$(MAKEFILE_DIR).[docs]'


# Docs
docs-serve:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
	mkdocs serve -a localhost:8000

docs-build:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
	mkdocs build

docs-deploy:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
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
	black $(MAKEFILE_DIR).

check-format:
	black --check $(MAKEFILE_DIR).

lint:
	ruff check $(MAKEFILE_DIR).


# Testing
test:
	pytest $(MAKEFILE_DIR)


# Cleanup
clean:
	rm -rf $(MAKEFILE_DIR)build \
	       $(MAKEFILE_DIR)dist \
	       $(MAKEFILE_DIR)*.egg-info \
	       $(MAKEFILE_DIR)__pycache__ \
	       $(MAKEFILE_DIR).pytest_cache \
	       $(MAKEFILE_DIR)docs/examples/example_data/Output \
	       $(MAKEFILE_DIR)site
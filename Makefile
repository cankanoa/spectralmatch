MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ENV_NAME = spectralmatch
GENERATE_HEADERS=python spectralmatch_qgis/generate_function_headers.py
QGISPLUGINNAME = spectralmatch_qgis

# Install
install:
	pip install $(MAKEFILE_DIR).

install-dev:
	pip install -e '$(MAKEFILE_DIR).[dev]'

install-docs:
	pip install -e '$(MAKEFILE_DIR).[docs]'

install-setup:
	bash -c "\
		conda create -y -n $(ENV_NAME) python>=3.10 gdal>=3.6 proj>=9.3 -c conda-forge && \
		source $$(conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(ENV_NAME) && \
		pip install . && \
		pip install -e '.[dev]' && \
		pip install -e '.[docs]' && \
		pre-commit install && \
		echo '✅ Setup complete. Environment \"$(ENV_NAME)\" is ready.' \
	"


# Docs
docs-serve:
	mkdir -p $(MAKEFILE_DIR)docs/images
	cp -r $(MAKEFILE_DIR)images/* $(MAKEFILE_DIR)docs/images/
	mkdocs serve -a localhost:8001

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

test-file:
	pytest $(path)

# Cleanup
clean:
	rm -rf $(MAKEFILE_DIR)build \
	       $(MAKEFILE_DIR)dist \
	       $(MAKEFILE_DIR)*.egg-info \
	       $(MAKEFILE_DIR)__pycache__ \
	       $(MAKEFILE_DIR).pytest_cache \
	       $(MAKEFILE_DIR)site \
	       $(MAKEFILE_DIR)spectralmatch_qgis/help/build \
	       $(MAKEFILE_DIR)spectralmatch_qgis/function_headers.json \
		   $(MAKEFILE_DIR)spectralmatch_qgis.zip
	find $(MAKEFILE_DIR)docs/examples/data_landsat -mindepth 1 ! -path "*/Input*" -exec rm -rf {} +
	find $(MAKEFILE_DIR)docs/examples/data_worldview -mindepth 1 ! -path "*/Input*" -exec rm -rf {} +


# QGIS

qgis-headers:
	@echo "Generating function_headers.json..."
	PYTHONPATH=. $(GENERATE_HEADERS)

qgis-build: qgis-headers
	@echo "---------------------------"
	@echo "Temporarily committing function_headers.json..."
	@echo "---------------------------"
	git add -f spectralmatch_qgis/function_headers.json
	git commit -m "temp: include function_headers.json in archive" --no-verify

	@echo "---------------------------"
	@echo "Creating plugin zip with function_headers.json..."
	@echo "---------------------------"
	git archive --prefix=$(QGISPLUGINNAME)/ -o $(QGISPLUGINNAME).zip HEAD:spectralmatch_qgis

	@echo "---------------------------"
	@echo "Cleaning up temp commit..."
	@echo "---------------------------"
	git reset --soft HEAD~1
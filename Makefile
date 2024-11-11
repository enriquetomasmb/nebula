POETRY_HOME := $(CURDIR)/.poetry
POETRY := $(POETRY_HOME)/bin/poetry

MIN_PYTHON_VERSION := 3.10

PYTHON_VERSIONS := 3.11 3.10

PYTHON := $(shell \
    for ver in $(PYTHON_VERSIONS); do \
        if command -v python$$ver >/dev/null 2>&1; then echo python$$ver; exit 0; fi; \
    done \
)

ifndef PYTHON
$(error "Python version $(MIN_PYTHON_VERSION) or higher is required but not found.")
endif

.PHONY: pre-install
pre-install:
	@echo "🐍 Using Python interpreter: $(PYTHON)"
	@echo "🐍 Checking if Python is installed"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo >&2 "$(PYTHON) is not installed. Aborting."; exit 1; }
	@echo "🐍 Checking Python version"
	@$(PYTHON) --version | grep -E "Python 3\.(1[0-9]|[2-9][0-9])" >/dev/null 2>&1 || { echo >&2 "Python $(MIN_PYTHON_VERSION) or higher is required. Aborting."; exit 1; }
	@echo "📦 Checking if Poetry is installed"
	@if ! command -v poetry >/dev/null 2>&1 || [ ! -d "$(POETRY_HOME)" ]; then \
	    echo "Poetry is not installed or POETRY_HOME does not exist. Installing Poetry."; \
	    curl -sSL https://install.python-poetry.org | POETRY_HOME=$(POETRY_HOME) $(PYTHON) -; \
	fi
	@echo "📦 Configuring Poetry"
	@if [ -z "$$CONDA_PREFIX" ] && [ -z "$$VIRTUAL_ENV" ]; then \
	    echo "Configuring Poetry to create a virtual environment."; \
	    $(POETRY) config virtualenvs.in-project true; \
	else \
	    echo "Configuring Poetry to use the existing environment."; \
	    $(POETRY) config virtualenvs.create false; \
	fi
	@echo "📦 Setting Poetry to use $(PYTHON)"
	@$(POETRY) env use $(PYTHON) || { echo "Failed to set Python version for Poetry. Aborting."; exit 1; }

.PHONY: install
install: pre-install ## Install the poetry environment and install the pre-commit hooks
	@echo "📦 Installing dependencies with Poetry"
	@PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring $(POETRY) install --with core
	@echo "🔧 Installing pre-commit hooks"
	@$(POETRY) run pre-commit install
	@$(MAKE) shell

.PHONY: full-install
full-install: pre-install ## Install the poetry environment and install the pre-commit hooks
	@echo "📦 Installing dependencies with Poetry"
	@PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring $(POETRY) install --with core,docs,dev
	@echo "🔧 Installing pre-commit hooks"
	@$(POETRY) run pre-commit install
	@$(MAKE) shell

.PHONY: shell
shell: ## Start a shell in the poetry environment
	@if [ -z "$$CONDA_PREFIX" ] && [ -z "$$VIRTUAL_ENV" ]; then \
	    echo "🐚 Activating virtual environment"; \
	    $(POETRY) shell; \
	else \
	    echo "🐚 Conda or virtual environment detected, skipping Poetry shell activation"; \
	fi

.PHONY: sync
sync: ## Sync the lock file
	@echo "📦 Syncing the lock file"
	@PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring $(POETRY) lock

.PHONY: update-libs
update-libs: ## Update libraries to the latest version
	@echo "🔧 This will override the version of current libraries. Do you want to continue? (y/n)"
	@read ans && [ $${ans:-N} = y ] || { echo "Update cancelled."; exit 1; }
	@echo "📦 Updating libraries..."
	@PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring $(POETRY) update

.PHONY: check
check: ## Run code quality tools.
	@echo "🛠️ Running code quality checks"
	@echo "🔍 Checking Poetry lock file consistency"
	@$(POETRY) check --lock
	@echo "🚨 Linting code with pre-commit"
	@$(POETRY) run pre-commit run -a

.PHONY: check-plus
check-plus: check ## Run additional code quality tools.
	@echo "🔍 Checking code formatting with black
	@$(POETRY) run black --check ."
	@echo "⚙️ Static type checking with mypy"
	@$(POETRY) run mypy
	@echo "🔎 Checking for obsolete dependencies"
	@$(POETRY) run deptry .

.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@$(POETRY) build

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: publish
publish: ## publish a release to pypi.
	@echo "🚀 Publishing: Dry run."
	@$(POETRY) config pypi-token.pypi $(PYPI_TOKEN)
	@$(POETRY) publish --dry-run
	@echo "🚀 Publishing."
	@$(POETRY) publish

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: doc-test
doc-test: ## Test if documentation can be built without warnings or errors
	@$(POETRY) run mkdocs build -f docs/mkdocs.yml -d _build -s

.PHONY: doc-build
doc-build: ## Build the documentation
	@$(POETRY) run mkdocs build -f docs/mkdocs.yml -d _build

.PHONY: doc-serve
doc-serve: ## Build and serve the documentation
	@$(POETRY) run mkdocs serve -f docs/mkdocs.yml

.PHONY: format
format: ## Format code with black and isort
	@echo "🎨 Formatting code"
	@$(POETRY) run black .
	@$(POETRY) run isort .

.PHONY: clean
clean: clean-build ## Clean up build artifacts and cache files
	@echo "🧹 Cleaning up build artifacts and caches"
	@rm -rf __pycache__ */__pycache__ .mypy_cache

.PHONY: help
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "💡 \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

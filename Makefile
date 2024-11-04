.PHONY: install
install: ## Install the poetry environment and install the pre-commit hooks
	@echo "📦 Installing dependencies with Poetry"
	@poetry install
	@echo "🔧 Installing pre-commit hooks"
	@poetry run pre-commit install
	@echo "🐚 Activating virtual environment"
	@poetry shell

.PHONY: check
check: ## Run code quality tools.
	@echo "🛠️ Running code quality checks"
	@echo "🔍 Checking Poetry lock file consistency"
	@poetry check --lock
	@echo "🚨 Linting code with pre-commit"
	@poetry run pre-commit run -a

.PHONY: check-plus
check-plus: check ## Run additional code quality tools.
	@echo "🔍 Checking code formatting with black
	@poetry run black --check ."
	@echo "⚙️ Static type checking with mypy"
	@poetry run mypy
	@echo "🔎 Checking for obsolete dependencies"
	@poetry run deptry .

.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@poetry build

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: publish
publish: ## publish a release to pypi.
	@echo "🚀 Publishing: Dry run."
	@poetry config pypi-token.pypi $(PYPI_TOKEN)
	@poetry publish --dry-run
	@echo "🚀 Publishing."
	@poetry publish

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: format
format: ## Format code with black and isort
	@echo "🎨 Formatting code"
	@poetry run black .
	@poetry run isort .

.PHONY: clean
clean: clean-build ## Clean up build artifacts and cache files
	@echo "🧹 Cleaning up build artifacts and caches"
	@rm -rf __pycache__ */__pycache__ .mypy_cache

.PHONY: help
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "💡 \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

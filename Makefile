UV := uv
PYTHON_VERSION := 3.11
UV_INSTALL_SCRIPT := https://astral.sh/uv/install.sh

command_exists = $(shell command -v $(1) >/dev/null 2>&1 && echo true || echo false)

define install_uv
	@echo "📦 uv is not installed. Installing uv..."
	@curl -LsSf $(UV_INSTALL_SCRIPT) | sh
endef

.PHONY: check-uv
check-uv:		## Check and install uv if necessary
	@if command -v $(UV) >/dev/null 2>&1; then \
		echo "📦 uv is already installed."; \
	else \
		echo "📦 uv is not installed. Installing uv..."; \
		curl -LsSf $(UV_INSTALL_SCRIPT) | sh; \
	fi

.PHONY: install-python
install-python: check-uv	## Install Python with uv
	@echo "🐍 Installing Python $(PYTHON_VERSION) with uv"
	@$(UV) python install $(PYTHON_VERSION)
	@echo "🔧 Configuring Python $(PYTHON_VERSION) as the default Python version"
	@$(UV) python pin $(PYTHON_VERSION)

.PHONY: install
install: install-python		## Install core dependencies
	@echo "📦 Installing core dependencies with uv"
	@$(UV) sync --group core
	@echo "🔧 Installing pre-commit hooks"
	@$(UV) run pre-commit install
	@echo ""
	@echo "🐳 Building nebula-frontend docker image. Do you want to continue? (y/n)"
	@read ans && [ $${ans:-N} = y ] || { echo "Build cancelled."; exit 1; }
	@docker build -t nebula-frontend -f nebula/frontend/Dockerfile .
	@echo ""
	@echo "🐳 Building nebula-core docker image. Do you want to continue? (y/n)"
	@read ans && [ $${ans:-N} = y ] || { echo "Build cancelled."; exit 1; }
	@docker build -t nebula-core .
	@echo ""
	@$(MAKE) shell

.PHONY: full-install
full-install: install-python	## Install all dependencies (core, docs)
	@echo "📦 Installing all dependencies with uv"
	@$(UV) sync --group core --group docs
	@echo "🔧 Installing pre-commit hooks"
	@$(UV) run pre-commit install
	@$(MAKE) shell

.PHONY: shell
shell:				## Start a shell in the uv environment
	@echo "🐚 Starting a shell in the uv environment"
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "🐚 Already in a virtual environment: $$VIRTUAL_ENV"; \
	elif [ ! -d ".venv" ]; then \
		echo "❌ .venv directory not found. Running 'make install' to create it..."; \
		$(MAKE) install; \
	else \
		echo "🐚 Run the following command to activate the virtual environment:"; \
		echo ""; \
		echo '[Linux/MacOS]	\033[1;32msource .venv/bin/activate\033[0m'; \
		echo '[Windows]	\033[1;32m.venv\\bin\\activate\033[0m'; \
		echo ""; \
		echo "🚀 NEBULA is ready to use!"; \
		echo "🚀 Created by \033[1;34mEnrique Tomás Martínez Beltrán\033[0m <\033[1;34menriquetomas@um.es\033[0m>"; \
	fi

.PHONY: lock
lock:				## Update the lock file
	@echo "🔒 This will update the lock file. Do you want to continue? (y/n)"
	@read ans && [ $${ans:-N} = y ] || { echo "Lock cancelled."; exit 1; }
	@echo "🔒 Locking dependencies..."
	@$(UV) lock

.PHONY: update-libs
update-libs:			## Update libraries to the latest version
	@echo "🔧 This will override the versions of current libraries. Do you want to continue? (y/n)"
	@read ans && [ $${ans:-N} = y ] || { echo "Update cancelled."; exit 1; }
	@echo "📦 Updating libraries..."
	@$(UV) update

.PHONY: check
check:				## Run code quality tools
	@echo "🛠️ Running code quality checks"
	@echo "🔍 Checking uv lock file consistency"
	@$(UV) sync
	@echo "🚨 Linting code with pre-commit"
	@$(UV) run pre-commit run -a

.PHONY: check-plus
check-plus: check		## Run additional code quality tools
	@echo "🔍 Checking code formatting with black"
	@$(UV) run black --check .
	@echo "⚙️ Static type checking with mypy"
	@$(UV) run mypy
	@echo "🔎 Checking for obsolete dependencies"
	@$(UV) run deptry .

.PHONY: build
build: clean-build		## Build the wheel file
	@echo "🚀 Creating wheel file"
	@$(UV) build

.PHONY: clean-build
clean-build:			## Clean build artifacts
	@rm -rf dist

.PHONY: publish
publish:			## Publish a release to PyPI
	@echo "🚀 Publishing...""
	@$(UV) publish --token $(PYPI_TOKEN)

.PHONY: build-and-publish
build-and-publish: build publish	## Build and publish the package

.PHONY: doc-test
doc-test:			## Test if documentation can be built without errors
	@$(UV) run mkdocs build -f docs/mkdocs.yml -d _build -s

.PHONY: doc-build
doc-build:			## Build the documentation
	@$(UV) run mkdocs build -f docs/mkdocs.yml -d _build

.PHONY: doc-serve
doc-serve:			## Serve the documentation locally
	@$(UV) run mkdocs serve -f docs/mkdocs.yml

.PHONY: format
format:				## Format code with black and isort
	@echo "🎨 Formatting code"
	@$(UV) run black .
	@$(UV) run isort .

.PHONY: clean
clean: clean-build		## Clean up build artifacts and caches
	@echo "🧹 Cleaning up build artifacts and caches"
	@rm -rf __pycache__ */__pycache__ .mypy_cache

.PHONY: help
help:				## Display available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "💡 \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

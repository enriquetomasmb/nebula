.PHONY: test
test: ## Test if documentation can be built without warnings or errors
	@poetry run mkdocs build -s

.PHONY: build
build: ## Build the documentation
	@poetry run mkdocs build -d _build

.PHONY: serve
serve: ## Build and serve the documentation
	@poetry run mkdocs serve

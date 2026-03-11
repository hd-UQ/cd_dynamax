.PHONY: lint test

clean:
	ruff format --exclude cd_dynamax/dynamax --exclude "**/*.ipynb" .
	ruff check --exclude cd_dynamax/dynamax --exclude "**/*.ipynb" . --fix

lint:
	ruff check . --exclude cd_dynamax/dynamax --exclude "**/*.ipynb"

test:
	pytest --ignore cd_dynamax/dynamax

build_docs:
	mkdocs build
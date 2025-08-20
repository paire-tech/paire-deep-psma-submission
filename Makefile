.DEFAULT_GOAL := help

SRC:=src
TESTS:=tests
CMD:=uv run

APP_NAME:=paire-deep-psma-submission
APP_VERSION:=$(shell uv version --short)-nnunetv2-ensemble-all2
APP_IMAGE:=$(APP_NAME):$(APP_VERSION)

-include .env
INPUT_FORMAT?=gc
INPUT_DIR?=./data/input
OUTPUT_DIR?=./data/output
WEIGHTS_DIR?=./weights
LOG_LEVEL?=INFO
LOG_FORMAT?=%(message)s
DEVICE?=auto

# Typing etc.

.PHONY: format
format: ## Format source code
	$(CMD) ruff format $(SRC) $(TESTS)

.PHONY: lint
lint: ## Lint source code
	$(CMD) ruff check $(SRC) $(TESTS)

.PHONY: lint-fix
lint-fix: ## Lint and fix source code
	$(CMD) ruff check $(SRC) $(TESTS) --fix

.PHONY: isort
isort: ## Sort imports using Ruff
	$(CMD) ruff check $(SRC) $(TESTS) --fix --select I

.PHONY: type
type: ## Type in source code
	$(CMD) mypy $(SRC) $(TESTS)

# Tests

.PHONY: test
test: ## Run unit tests
	$(CMD) pytest $(TESTS) -vv

# Docker

.PHONY: build
build: ## Build a production docker image
	docker build --progress=plain -t $(APP_NAME):$(APP_VERSION) .

.PHONY: run
run: ## Run the production docker image
	docker run --rm -it \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		-v $(INPUT_DIR):/input/:ro \
		-v $(OUTPUT_DIR):/output/ \
		-e INPUT_FORMAT=$(INPUT_FORMAT) \
		-e LOG_LEVEL=$(LOG_LEVEL) \
		$(APP_IMAGE)

.PHONY: export
export: ## Export the production docker image
	docker save $(APP_NAME):$(APP_VERSION) | gzip > $(APP_NAME)-v$(subst .,-,$(APP_VERSION)).tar.gz

# Misc

.PHONY: all
all: format type lint isort test-cov ## Run all formatting commands

.PHONY: clean
clean: ## Clear local caches and build artifacts
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -rf coverage.xml
	rm -rf *.log

.PHONY: help
help: ## Show available commands
	@echo "Available targets:"
	@awk '/^[a-zA-Z0-9_-]+:.*?## .*$$/ { \
		helpCommand = substr($$0, 1, index($$0, ":")-1); \
		helpMessage = substr($$0, index($$0, "## ") + 3); \
		printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
	}' $(MAKEFILE_LIST)

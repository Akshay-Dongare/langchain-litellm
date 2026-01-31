.PHONY: all format lint test tests integration_tests docker_tests help extended_tests lint_imports check_imports lock_fix

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

######################
# LOCKING AND SETUP
######################

# NEW: Refreshes the lock file to match pyproject.toml
lock_fix:
	poetry lock

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache

# 1. Inline Lint Imports (Checks for forbidden dependencies)
lint_imports:
	@echo "Checking for forbidden imports..."
	@! git grep -nE "^from langchain\." . || (echo "Error: Importing directly from 'langchain' is forbidden." && exit 1)
	@! git grep -nE "^from langchain_experimental\." . || (echo "Error: Importing directly from 'langchain_experimental' is forbidden." && exit 1)
	@! git grep -nE "^from langchain_community\." . || (echo "Error: Importing directly from 'langchain_community' is forbidden." && exit 1)

# 2. Inline Check Imports (Verifies module loading)
check_imports:
	@echo "Verifying module imports..."
	@poetry run python -c 'import sys, traceback; from importlib.machinery import SourceFileLoader; \
	files = sys.argv[1:]; has_fail=False; \
	for f in files: \
		try: SourceFileLoader("x", f).load_module() \
		except: print(f"❌ FAILED to load: {f}"); traceback.print_exc(); has_fail=True; \
	sys.exit(1 if has_fail else 0)' \
	$(shell find langchain_litellm -name '*.py')

# Configuration for specific targets
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/litellm --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_litellm
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

# Main lint target (Runs imports check + ruff check + ruff format check + mypy)
lint lint_diff lint_package lint_tests: lint_imports
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

# Format target (Runs ruff format + ruff fix)
format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I --fix $(PYTHON_FILES)

# Spell Check Targets
spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'lock_fix                     - refresh poetry.lock file'
	@echo 'check_imports                - check imports'
	@echo 'lint_imports                 - check for forbidden imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'spell_check                  - run spell checker'
	@echo 'spell_fix                    - run spell checker and auto-fix typos'
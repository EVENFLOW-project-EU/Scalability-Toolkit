# Default target, executed when typing 'make'
.DEFAULT_GOAL := help

COLOR1 := \033[36m
COLOR2 := \033[1
COLOR_NORMAL := \033[0m

.PHONY: help
help:	## Show this help message
	@echo "${COLOR1}weather-analytics${COLOR_NORMAL}"
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make ${COLOR1}<target>${COLOR_NORMAL}\n"}\
 	/^[a-zA-Z_-]+:.*?##/ { printf "  ${COLOR1}%-10s${COLOR_NORMAL} %s\n", $$1, $$2 }\
  	/^##@/ { printf "\n${COLOR2}m%s${COLOR_NORMAL}\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


# Formatting and component style
FLAKE_IGNORES := W503,E402,E731,E203,W291,E712,F631	# See README for a detailed explanation
FLAKE_EXCLUSIONS := .git,.venv,.vscode,.idea,.pytest_cache
FLAKE_EXTRAS := --jobs 1 --max-line-length 120

.PHONY: black isort flake reformat
black:	## Applies black formatter in the Python codebase
	black --line-length 120 --verbose ./evenflow/**
	black --line-length 120 --verbose ./evenflow/**
	@echo 'black finished successfully.'

isort:	## Applies isort formatter in import statements in the Python codebase
	isort -v --profile black ./evenflow/**
	@echo 'isort finished successfully.'

flake:	## Python linting quality control
	flake8 ./evenflow/** ${FLAKE_EXTRAS} --ignore ${FLAKE_IGNORES} --exclude ${FLAKE_EXCLUSIONS} --max-line-length 120 -v
	@echo 'Flake finished successfully.'

reformat: isort black flake	## Run isort, black and flake targets.
	@echo "DONE!"

.PHONY: test
test:	## Runs all available tests and fails on first error
	pytest evenflow/tests --disable-pytest-warnings
	@echo 'All tests completed without error'

.PHONY: pre-commit
pre-commit: reformat test clean

.PHONY: coverage-report, coverage-html coverage-xml
.coverage: 	## Coverage report for the python codebase, will create `.coverage` file.
	coverage run --source=./evenflow/** --omit="*/tests*" -m pytest --exitfirst -v tests

coverage-report: .coverage	## Coverage report for the python codebase in the CLI.
	coverage report -m

coverage-html: .coverage	## Coverage report for the python codebase in an HTML format under `htmlcov`
	coverage html

coverage-xml: .coverage		## Coverage report for the python codebase in an ΧΜL format in `coverage.xml` (top level).
	coverage xml -i

# Remove project temporary and cache related files
.PHONY: clean-coverage clean-pyc clean-resources clean

clean: clean-pyc clean-coverage	clean-resources	## Delete all project generated files.

clean-pyc:	## Delete python cache related files.
	rm -rf .pytest_cache
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '__pycache__*' -exec rm -fr {} +

clean-coverage:	## Delete python coverage related files.
	rm -f .coverage
	rm -rf htmlcov
	rm -f coverage.xml

clean-resources:	## Delete python logs related files.
	rm -rf logs
	rm -rf metrics
	rm -rf kafka_2.13-3.1.2
	rm -rf SUSY.csv.gz


# Docs
.PHONY: docs docs-clean

docs:	## Updates the 'docs' folder with documentation generated from docstrings.
	pdoc ./evenflow/** -o docs

docs-clean:
	rm -rf docs


# Download
.PHONY: download-kafka, download-dataset, download

download-kafka:	## Download kafka
	curl -sSOL https://dlcdn.apache.org/kafka/3.1.2/kafka_2.13-3.1.2.tgz
	tar -xzf kafka_2.13-3.1.2.tgz
	rm kafka_2.13-3.1.2.tgz

download-datasets:
	curl -sSOL https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz

download: download-kafka download-datasets	## Download kafka and datasets

# Env creation
.PHONY: install, install-java, create-requirements
	
install-java:	## Install java
	sudo apt install openjdk-11-jdk

install:	## Install the project in editable mode.
	conda env create -n evenflow-1 -f conda.yaml
	conda activate evenflow-1

create-requirements:	## Create requirements.txt file from conda environment
	pip list --format=freeze > requirements.tx

# SETUP everything
.PHONY: setup

setup: install-java install download-kafka create-requirements download-kafka download-datasets
	echo "Setup completed successfully."


SHELL := /bin/bash

.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and python3 artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -path ./venv -prune -o -name '*.egg-info' -exec rm -fr {} +
	find . -path ./venv -prune -o -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove python3 file artifacts
	find . -path ./venv -prune -o -name '*.pyc' -exec rm -f {} +
	find . -path ./venv -prune -o -name '*.pyo' -exec rm -f {} +
	find . -path ./venv -prune -o -name '*~' -exec rm -f {} +
	find . -path ./venv -prune -o -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 satsim tests

test: ## run tests quickly with the default python3
	source gpu_select.sh && py.test

testp: ## run tests and don't capture prints
	source gpu_select.sh && py.test -s

test-all: ## run tests on every python3 version with tox
	tox

coverage: ## check code coverage quickly with the default python3
	source gpu_select.sh && coverage run --source satsim -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

cpucoverage: ## check code coverage quickly with the default python3
	CUDA_VISIBLE_DEVICES="" && coverage run --source satsim -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/satsim.rst
	rm -f docs/modules.rst
	rm -fr docs/api
	sphinx-apidoc -fMT satsim -o docs/api
	#sphinx-apidoc -o docs/ satsim
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean docs ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	ls -l dist

develop: clean
	python3 -m pip install -U -r requirements_dev.txt --user
	python3 setup.py develop --user
	@echo  IMPORTANT: You may need to close and restart your shell after running "make develop".

undevelop: clean
	python3 setup.py develop --user --uninstall
	rm ~/.local/bin/satsim

install: clean ## install the package to the active python3's site-packages
	python3 setup.py install --record .install.log
	@echo  IMPORTANT: You may need to close and restart your shell after running "make install".

docker: docs dist
	docker build -t satsim:0.18.0 -t satsim:latest -f docker/ubuntu20.04_cuda11.2_py3.8.dockerfile .

dind:
	docker run --rm -it -v .:/workspace/ -w /workspace python:3.8-bullseye ./build.sh
	docker build -t satsim:0.18.0-cuda11.2 -f docker/ubuntu20.04_cuda11.2_py3.8.dockerfile .
	docker build -t satsim:0.18.0-cuda11.8 -t satsim:0.18.0 -t satsim:latest -f docker/ubuntu22.04_cuda11.8_py3.10.dockerfile .

uninstall: clean
	cat .install.log | xargs rm -rf

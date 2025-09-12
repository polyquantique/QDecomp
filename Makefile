# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# Detect the operating system
ifeq ($(OS), Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

# Python executable
ifeq ($(DETECTED_OS), Windows)
	PYTHON := $(shell where py 2>NUL || where py3 2>NUL || where python 2>NUL || where python3 2>NUL)
	OPEN := cmd /c start
	RM := del /q
else
	PYTHON := $(shell which py 2>/dev/null || which py3 2>/dev/null || which python 2>/dev/null || which python3 2>/dev/null)
	OPEN := open
	RM := rm -rf
endif


# Directories
SRC_DIR := src
DOCS_DIR := docs
TEST_DIR := tests


# Generate the documentation
.PHONY: docs
docs:
	$(MAKE) -C $(DOCS_DIR) figures
	$(MAKE) -C $(DOCS_DIR) html


# Format the code with black and isort
.PHONY: format
format:
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR)
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR) -l 100


# Run the tests
.PHONY: test
test:
	$(PYTHON) -m pytest $(TEST_DIR)

# Run the tests with coverage
.PHONY: test_cov
test_cov:
	$(PYTHON) -m pytest --cov=$(SRC_DIR) $(TEST_DIR) --cov-report=html --cov-branch

# Show the test coverage report
.PHONY: test_report
test_report:
	$(OPEN) .\htmlcov\index.html


# Clean the repository
.PHONY: clean
clean:
	$(RM) .\htmlcov

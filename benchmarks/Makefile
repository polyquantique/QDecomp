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

# Set platform-specific commands
ifeq ($(DETECTED_OS), Windows)
	PYTHON := $(shell where py 2>NUL || where py3 2>NUL || where python 2>NUL || where python3 2>NUL)
	OPEN := cmd /c start
else
	PYTHON := $(shell which py 2>/dev/null || which py3 2>/dev/null || which python 2>/dev/null || which python3 2>/dev/null)
	OPEN := open
endif


# Useful variables
DATA_DIR := data
GRAPH_DIR := graphs

DEFAULT_PROFILE := rz_decomp-angle1_8325957145940461-epsilon0_001.prof
VERSION := $(shell "$(PYTHON)" -c "import qdecomp; print(qdecomp.__version__.replace('.', '_'))")


# Set the default goal
.DEFAULT_GOAL := help


# Show help message
.PHONY: help
help:
	@echo Available targets:
	@echo $(VERSION)
	@echo   run_missing_versions   - Run benchmarks for specified package versions.
	@echo   run_current_version    - Run benchmarks for the current package version.
	@echo   rerun_all              - Rerun all benchmarks, even if profiles already exist.
	@echo   show_graphs            - Plot profiling results and show them.
	@echo   show_profiling         - Show the profiling graph for a default set of parameters.


# Run benchmarks for missing package versions
.PHONY: run_missing_versions
run_missing_versions:
	$(PYTHON) -c "from run_benchmarks import run_all_benchmarks; run_all_benchmarks()"

# Run benchmarks for current package version
.PHONY: run_current_version
run_current_version:
	$(PYTHON) -c "from run_benchmarks import run_single_benchmark; import qdecomp as qd; run_single_benchmark(qd.__version__)"

# Rerun all the benchmarks
.PHONY: rerun_all
rerun_all:
	$(PYTHON) -c "from run_benchmarks import run_all_benchmarks; run_all_benchmarks(rerun=True)"


# Plot profiling results
.PHONY: show_graphs
show_graphs:
	$(PYTHON) -c "import benchmark_utils as util; util.plot_version_profiling()"
	$(PYTHON) -c "import benchmark_utils as util; util.plot_epsilon_profiling()"
	$(OPEN) $(GRAPH_DIR)/version_profiling.svg
	$(OPEN) $(GRAPH_DIR)/epsilon_profiling.svg

# Show the profiling graph for a default set of parameters
.PHONY: show_profiling
show_profiling:
	$(PYTHON) -m snakeviz $(DATA_DIR)/v$(VERSION)/$(DEFAULT_PROFILE)

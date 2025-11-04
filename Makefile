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
	RMFILE := del /q
    RMDIR := rmdir /s /q
else
	PYTHON := $(shell which py 2>/dev/null || which py3 2>/dev/null || which python 2>/dev/null || which python3 2>/dev/null)
	OPEN := open
	RMFILE := rm -f
    RMDIR := rm -rf
endif


# C++ compiler settings
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pthread


# Directories
SRC_DIR := src
DOCS_DIR := docs
TEST_DIR := tests
PY_TEST_DIR := $(TEST_DIR)/python
CPP_TEST_DIR := $(TEST_DIR)/cpp
LIBS_DIR := libs
GTEST_DIR := $(LIBS_DIR)/googletest
RZ_APPROX_DLL_FILE := $(SRC_DIR)/qdecomp/utils/grid_problem/cpp/lib_rz_approx.dll


# For C++ tests
CPP_TEST_MAIN := $(CPP_TEST_DIR)/test_main
ifeq ($(DETECTED_OS), Windows)
	CPP_TEST_SRC_FILES := $(shell powershell -Command "Get-ChildItem -Path '$(CPP_TEST_DIR)' -Recurse -Include *.cpp -Depth 2 | ForEach-Object { $$_.FullName }")
else
	CPP_TEST_SRC_FILES := $(shell find $(CPP_TEST_DIR) -name '*.cpp')
endif


# Set the default goal
.DEFAULT_GOAL := help


# Show available targets  
.PHONY: help
help:
	@echo Available targets:
	@echo   docs            - Build documentation
	@echo   format          - Format source and test code using isort and black
	@echo   test            - Run tests
	@echo   test_cov        - Run tests with coverage
	@echo   test_report     - Open coverage report
	@echo   rz_approx_dll   - Compile the lib_rz_approx dynamic library
	@echo   compile_gtest   - Compile the googletest library
	@echo   test_cpp        - Compile and run C++ tests
	@echo   clean           - Remove coverage artifacts


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


# Run the Python tests
.PHONY: test
test:
	$(PYTHON) -m pytest $(PY_TEST_DIR) -n auto

# Run the tests with coverage
.PHONY: test_cov
test_cov:
	$(PYTHON) -m pytest --cov=$(SRC_DIR) $(PY_TEST_DIR) --cov-report=html --cov-branch -n auto

# Show the test coverage report
.PHONY: test_report
test_report:
	$(OPEN) ./htmlcov/index.html


# Compile and execute a .exe file from a .cpp files
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -I $(LIBS_DIR) -I $(SRC_DIR) -o $@
	./$@

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -I $(LIBS_DIR) -I $(SRC_DIR) -o $@

# Compile dynamic libraries
%.dll: %.o
	$(CXX) -shared -fPIC -lstdc++ $(CXXFLAGS) -o $@ $< -static


# Compile the lib_rz_approx dynamic library
.PHONY: rz_approx_dll
rz_approx_dll: $(RZ_APPROX_DLL_FILE)


# Compile the googletest library
.PHONY: compile_gtest
compile_gtest:
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/googletest -I $(GTEST_DIR)/googletest/include -c $(GTEST_DIR)/googletest/src/gtest-all.cc -o $(GTEST_DIR)/googletest/gtest-all.o
	ar rcs $(GTEST_DIR)/googletest/libgtest.a $(GTEST_DIR)/googletest/gtest-all.o

# Run the C++ tests
.PHONY: test_cpp
test_cpp:
	$(CXX) $(CXXFLAGS) $(CPP_TEST_SRC_FILES) -I $(GTEST_DIR)/googletest/include -I $(LIBS_DIR) -I $(SRC_DIR) -L $(GTEST_DIR)/googletest -lgtest -o $(CPP_TEST_DIR)/test_main
	$(CPP_TEST_DIR)/test_main


# Clean the repository
.PHONY: clean
clean:
	-$(RMDIR) htmlcov
	-$(RMFILE) $(CPP_TEST_DIR)/test_main
	-$(RMFILE) $(CPP_TEST_DIR)/test_main.exe
	-$(RMFILE) $(RZ_APPROX_DLL_FILE)

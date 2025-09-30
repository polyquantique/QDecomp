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
	RMFILES := del /q
    RMDIR := rmdir /s /q
else
	RMFILES := rm -f
    RMDIR := rm -rf
endif


# Makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# LaTeX figure generation
SOURCE_DIR = source
STATIC_DIR = source/_static
FIGURES = package_structure
FIGURES := $(addprefix $(STATIC_DIR)/, $(FIGURES))


# Set the default goal
.DEFAULT_GOAL := help


# Show help message
.PHONY: help
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
.PHONY: makefile
%: makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: figures
figures: $(FIGURES:=.svg)

%.dvi: %.tex
	latex -output-format=dvi -output-directory=$(STATIC_DIR) $<
	del /q $*.aux $*.log

%.svg: %.dvi
	dvisvgm -n -o $@ $<

.PHONY: clean
clean:
	$(RMFILES) ${FIGURES}.dvi ${FIGURES}.svg
	$(RMFILES) ${FIGURES}.aux ${FIGURES}.log ${FIGURES}.fdb_latexmk ${FIGURES}.fls ${FIGURES}.pdf ${FIGURES}.synctex.gz

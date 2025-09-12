# Makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# LaTeX figure generation
SOURCE_DIR = source
STATIC_DIR = source\_static
FIGURES = package_structure
FIGURES := $(addprefix $(STATIC_DIR)\, $(FIGURES))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
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
	del /q ${FIGURES}.dvi ${FIGURES}.svg
	del /q ${FIGURES}.aux ${FIGURES}.log ${FIGURES}.fdb_latexmk ${FIGURES}.fls ${FIGURES}.pdf ${FIGURES}.synctex.gz

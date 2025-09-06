# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import qdecomp
project = "QDecomp"
copyright = "2025, Vincent Girouard, Olivier Romain, Marius Trudeau, Francis Blais"
author = "Vincent Girouard, Olivier Romain, Marius Trudeau, Francis Blais"
release = qdecomp.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Generates documentation from docstrings
    "sphinx.ext.napoleon",  # Supports Google and NumPy docstring formats
    "sphinx.ext.viewcode",  # Links source code in documentation
    "sphinx.ext.todo",  # Support for TODO notes
    "sphinx.ext.autosummary",  # Generates summary tables for modules, classes, and functions
    "sphinxcontrib.bibtex",  # Bibliography support
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_default_options = {"special-members": "__init__", "no-value": True}
autodoc_member_order = "bysource"  # Order members by source code order
html_static_path = ['_static']
bibtex_bibfiles = ["references.bib"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"
suppress_warnings = ["bibtex.duplicate_citation"]

autosummary_generate = True  # Automatically create summary tables

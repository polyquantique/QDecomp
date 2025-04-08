# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "QDecomp"
copyright = "2025, Vincent Girouard, Olivier Romain, Marius Trudeau, Francis Blais"
author = "Vincent Girouard, Olivier Romain, Marius Trudeau, Francis Blais"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Generates documentation from docstrings
    "sphinx.ext.napoleon",  # Supports Google and NumPy docstring formats
    "sphinx.ext.viewcode",  # Links source code in documentation
    "sphinx.ext.todo",  # Support for TODO notes
    "sphinx.ext.autosummary",  # Generates summary tables for modules, classes, and functions
]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_default_options = {"special-members": "__init__"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

autosummary_generate = True  # Automatically create summary tables

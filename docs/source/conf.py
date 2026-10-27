# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import sys
# Aliased: a bare `version` name would be picked up as Sphinx's own `version`
# config value, which must be a string.
from importlib.metadata import version as _get_version
sys.path.insert(0, os.path.abspath('../../')) # Points to project root

project = 'Fast-Select'
copyright = '2025-2026, Gavin Lynch'
author = 'Gavin Lynch'
release = _get_version('fast-select')
version = release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_copybutton',
]
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


def _literal_selector_attributes(app, what, name, obj, options, lines):
    """Render fitted attribute names literally in the new selector docstrings."""
    if name.startswith(('fast_select.GiniIndex.', 'fast_select.JointInformation.')):
        for i, line in enumerate(lines):
            lines[i] = re.sub(r'(?<![\w`])([A-Za-z]\w*_)(?![\w`])', r'``\1``', line)


def setup(app):
    app.connect('autodoc-process-docstring', _literal_selector_attributes)

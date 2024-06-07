# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import subprocess

subprocess.call(['sphinx-apidoc', '-f', '-o', '.', '../nebula'])  # The same that 'make apidoc'

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NEBULA'
copyright = '2024, Enrique Tomás Martínez Beltrán'
author = 'Enrique Tomás Martínez Beltrán'
# The short X.Y version
version = '1.0'
# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.napoleon',
#     # 'sphinx.ext.viewcode',
#     'autoapi.extension'
# ]
extensions = [
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'autoapi.extension',  # Automatically generate API documentation
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinx.ext.todo',  # Support for todo items
    'sphinx.ext.autodoc',  # Support for automatic documentation
    'sphinx.ext.autosummary',  # Support for automatic summaries
    'sphinx.ext.doctest',  # Support for doctests
]


autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'app', 'setup.py', 'docs']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "repository_url": "https://github.com/enriquetomasmb/nebula",
    "use_repository_button": True,
}

html_logo = "_static/nebula-logo.jpg"

html_title = "NEBULA Documentation"

# -- Options for autoapi extension -------------------------------------------
autoapi_root = 'api'
autoapi_template_dir = "_templates/autoapi"
autoapi_dirs = ['../nebula']
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
# autoapi_keep_files = True


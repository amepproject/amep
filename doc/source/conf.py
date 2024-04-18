"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""
project = 'AMEP'
copyright = '2023-2024, Lukas Hecht, Kay-Robert Dormann, Kai Luca Spanheimer'
author = 'Lukas Hecht, Kay-Robert Dormann, Kai Luca Spanheimer'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list = [
        "sphinx.ext.napoleon",
        "sphinx.ext.mathjax",
        "myst_parser",
        "sphinx.ext.duration",
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx_new_tab_link"
]

templates_path: list = ['_templates']
exclude_patterns: list = []
napoleon_numpy_docstring = True
autosummary_generate = True
autosummary_filename_map = {
        "amep.functions.Gaussian": "amep.functions.GaussianClass.rst",
        "amep.functions.Gaussian2d": "amep.functions.Gaussian2dClass.rst"
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_logo = "_static/images/amep-logo_v2.png"
html_favicon = "_static/images/amep-logo_v2.png"

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
        # "use_edit_page_button": True
        "icon_links": [
            {
                # Label for this link
                "name": "GitHub",
                # URL where the link will redirect
                "url": "https://github.com/amepproject/amep",  # required
                # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
                "icon": "fa-brands fa-square-github",
                # The type of image to be used (see below for details)
                "type": "fontawesome",
            },
            {
                # Label for this link
                "name": "PyPI",
                # URL where the link will redirect
                "url": "https://pypi.org/project/amep/",  # required
                # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
                "icon": "fa-brands fa-python",
                # The type of image to be used (see below for details)
                "type": "fontawesome",
            }
            ]
        }

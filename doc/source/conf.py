"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""
project = 'AMEP'
copyright = '2023-2025, Lukas Hecht, Kay-Robert Dormann, Kai Luca Spanheimer'
author = 'Lukas Hecht, Kay-Robert Dormann, Kai Luca Spanheimer'
release = '1.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list = [
        "sphinx.ext.napoleon",
        "sphinx.ext.mathjax",
        "myst_parser",
        "sphinx.ext.duration",
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx_new_tab_link",
        # 'versionwarning.extension',
        # "notfound.extension",
]

templates_path: list = ['_templates']
exclude_patterns: list = []
napoleon_numpy_docstring = True
autosummary_generate = True
autosummary_filename_map = {
        "amep.functions.Gaussian": "amep.functions.GaussianClass.rst",
        "amep.functions.Gaussian2d": "amep.functions.Gaussian2dClass.rst"
}
# notfound_urls_prefix = "/"+release+"/"
# notfound_urls_prefix = None
# notfound_template = "404.rst"
# notfound_template = "custom-class-template.rst"

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
                # The type of image to be used
                "type": "fontawesome",
            },
            {
                # Label for this link
                "name": "PyPI",
                # URL where the link will redirect
                "url": "https://pypi.org/project/amep/",  # required
                # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
                "icon": "fa-brands fa-python",
                # The type of image to be used
                "type": "fontawesome",
            }
        ],
        "switcher": {
           "json_url": "https://amepproject.de/switcher.json",
           "version_match": release
        },
        "check_switcher": True,
        "navbar_persistent": ["search-button.html", "theme-switcher.html"],
        "navbar_end": ["version-switcher.html", "icon-links.html"],
    }

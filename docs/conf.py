project = "llmSHAP"
author = "Filip Naudot"
extensions = [
    "myst_parser",            # Allow Markdown (for README).
    "sphinx.ext.autodoc",     # Pull docstrings from code.
    "sphinx.ext.napoleon",    # Google/NumPy style docstrings.
    "sphinx.ext.viewcode",    # Link to highlighted source.
    "sphinx.ext.autosummary", # Optional summaries.
    "sphinx_design",          # Allow for nicer documentation. 
]

# Treat README.md as Markdown via MyST.
myst_enable_extensions = ["colon_fence", "deflist"]

# Keep autodoc light.
autodoc_mock_imports = [
    "openai",
    "dotenv",
    "sentence_transformers",
]
autodoc_typehints = "description"
autosummary_generate = False

################################################
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))

# HTML theme.
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/filipnaudot/llmSHAP",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_source_button": True,
    "use_repository_button": True,
    # "use_issues_button": True,
    "use_edit_page_button": False,
}
html_static_path = ["_static"]

# Don not crash on minor nitpicks.
nitpicky = False
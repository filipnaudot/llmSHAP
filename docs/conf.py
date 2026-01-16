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

html_logo = "./_static/llmSHAP-logo-lightmode.png"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/filipnaudot/llmSHAP",
    "repository_branch": "main",
    "path_to_docs": "docs",

    "use_repository_button": True,
    "use_source_button": False,
    "use_edit_page_button": False,
    "use_download_button": False,
    "use_fullscreen_button": False,

    "logo": {
        "image_light": "./_static/llmSHAP-logo-lightmode.png",
        "image_dark": "./_static/llmSHAP-logo-darkmode.png",
        # "text": "llmSHAP",
        "alt_text": "llmSHAP documentation",
    }
}
html_static_path = ["_static"]

# Don not crash on minor nitpicks.
nitpicky = False
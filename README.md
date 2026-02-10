<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/filipnaudot/llmSHAP/main/docs/_static/llmSHAP-logo-lightmode.png">
        <img alt="llmSHAP logo" src="https://raw.githubusercontent.com/filipnaudot/llmSHAP/main/docs/_static/llmSHAP-logo-darkmode.png" width="50%" height="50%">
    </picture>
</div>
<br/>

![Unit Tests](https://github.com/filipnaudot/llmSHAP/actions/workflows/test.yml/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://filipnaudot.github.io/llmSHAP/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/llmshap?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=PyPI+downloads)](https://pepy.tech/projects/llmshap)

A multi-threaded explainability framework using Shapley values for LLM-based outputs.

---

## Getting started
Install the `llmshap` package (with all optional dependencies):
```bash
pip install "llmshap[full]"
```

Install in editable mode with all optional dependencies (after cloning the repository):
```bash
pip install -e ".[full]"
```

Documentation is available at [llmSHAP Docs](https://filipnaudot.github.io/llmSHAP/) and a hands-on tutorial can be found [here](https://filipnaudot.github.io/llmSHAP/tutorial.html).

- [Full documentation](https://filipnaudot.github.io/llmSHAP/)  
- [Tutorial](https://filipnaudot.github.io/llmSHAP/tutorial.html)

---

# Example usage

```python
from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
from llmSHAP.llm import OpenAIInterface

data = "In what city is the Eiffel Tower?"
handler = DataHandler(data, permanent_keys={0,3,4})
result = ShapleyAttribution(model=OpenAIInterface("gpt-4o-mini"),
                            data_handler=handler,
                            prompt_codec=BasicPromptCodec(system="Answer the question briefly."),
                            use_cache=True,
                            num_threads=16,
                            ).attribution()

print("\n\n### OUTPUT ###")
print(result.output)

print("\n\n### ATTRIBUTION ###")
print(result.attribution)

print("\n\n### HEATMAP ###")
print(result.render())
```

## Multimodal example with `Image`:
The following example shows `llmSHAP` with images. Note: `EmbeddingCosineSimilarity` downloads an embedding model on first use.
```python
from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, EmbeddingCosineSimilarity, Image
from llmSHAP.llm import OpenAIInterface

data = {
    "question": "Has our stockprice increased or decreased since the beginning?",
    "Num employees"        : "The company has about 450 employees.",
    "[IMAGE] Stock chart"  : Image(image_path="./docs/_static/demo-stock-price.png"),
    "Report release date"  : "Quarterly reports are released on the 15th.",
    "Headquarter Location" : "The headquarters is located in a mid-sized city.",
    "Num countries"        : "It has offices in three countries."
}

result = ShapleyAttribution(model=OpenAIInterface("gpt-5-mini", reasoning="low"),
                            data_handler=DataHandler(data, permanent_keys={"question"}),
                            prompt_codec=BasicPromptCodec(system="Answer the question briefly."),
                            use_cache=True,
                            num_threads=35,
                            value_function=EmbeddingCosineSimilarity(),
                            ).attribution()

print("\n\n### OUTPUT ###")
print(result.output)

print("\n\n### HEATMAP ###")
print(result.render(abs_values=True, render_labels=True))
```

<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/filipnaudot/llmSHAP/main/docs/_static/example-result-lightmode.png">
        <img alt="llmSHAP logo" src="https://raw.githubusercontent.com/filipnaudot/llmSHAP/main/docs/_static/example-result-darkmode.png" width="100%" height="100%">
    </picture>
</div>



---

## Example data

You can pass either a string or a dictionary:

```python
from llmSHAP import DataHandler

# String input
data = "The quick brown fox jumps over the lazy dog"
handler = DataHandler(data)

# Dictionary input
data = {"a": "The", "b": "quick", "c": "brown", "d": "fox"}
handler = DataHandler(data)
```

To exclude certain keys from the computations, use `permanent_keys`:
```python
from llmSHAP import DataHandler

data = {"a": "The", "b": "quick", "c": "brown", "d": "fox"}
handler = DataHandler(data, permanent_keys={"a", "d"})

# Get data with index 1 WITHOUT the permanent features.
print(handler.get_data({1}, exclude_permanent_keys=True, mask=False))
# Output: {'b': 'quick'}

# Get data with index 1 AND the permanent features.
print(handler.get_data({1}, exclude_permanent_keys=False, mask=False))
# Output: {'a': 'The', 'b': 'quick', 'd': 'fox'}
```
---


## Comparison with TokenSHAP
| Capability                                                                | **llmSHAP**                                                 | **TokenSHAP**                  |
| ------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------ |
| Threaded                                                                  | ✅ (optional ``num_threads``)                                | ❌                              |
| Modular architecture                                                      | ✅                                                           | ❌                              |
| Exact Shapley option                                                      | ✅ (Full enumeration)                                        | ❌ (Monte Carlo sampling)       |
| Generation caching across coalitions                                      | ✅                                                           | ❌                              |
| Heuristics                                                                | SlidingWindow • Monte Carlo • Counterfactual                 | Monte Carlo                    |
| Sentence-/chunk-level attribution                                         | ✅                                                           | ✅                             |
| Permanent context pinning (always-included features)                      | ✅                                                           | ❌                              |
| Pluggable similarity metric                                               | ✅ TF-IDF, embeddings                                        | ✅ TF-IDF, embeddings          |
| Docs & tutorial                                                           | ✅ Sphinx docs + tutorial                                    | ✅ README only                 |
| Unit tests & CI                                                           | ✅ Pytest + GitHub Actions                                   | ❌                              |
| Vision object attribution                                                 | ❌                                                           | ✅ PixelSHAP                   |

---

<br/>
<br/>
<br/>

# Stars ⭐️

[![Star History Chart](https://api.star-history.com/svg?repos=filipnaudot/llmSHAP&type=Date&legend=top-left)](https://www.star-history.com/#filipnaudot/llmSHAP&type=date&legend=top-left)
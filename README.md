<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="/docs/llmSHAP-logo-lightmode.png">
        <img alt="lighbench logo" src="/docs/llmSHAP-logo-darkmode.png" width="50%" height="50%">
    </picture>
</div>

# llmSHAP
![Unit Tests](https://github.com/filipnaudot/llmSHAP/actions/workflows/test.yml/badge.svg)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://filipnaudot.github.io/llmSHAP/)

Lightweight explainability framework using Shapley values for LLM-based outputs.

---

## Getting started

Install in editable mode with test dependencies:

```bash
pip install -e .[dev]  # for bash
```
```bash
pip install -e '.[dev]'  # for zsh
```

Documentation is available at [llmSHAP Docs](https://filipnaudot.github.io/llmSHAP/) and a hands-on tutorial can be found [here](https://filipnaudot.github.io/llmSHAP/tutorial.html).

- [Full documentation](https://filipnaudot.github.io/llmSHAP/)  
- [Tutorial](https://filipnaudot.github.io/llmSHAP/tutorial.html)

---

## Example usage

```python
from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
from llmSHAP.llm import OpenAIInterface

data = "In what city is the Eiffel Tower?"
handler = DataHandler(data, permanent_keys={0,3,4})
prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
llm = OpenAIInterface("gpt-4o-mini")

shap = ShapleyAttribution(model=llm, data_handler=handler, prompt_codec=prompt_codec, use_cache=True)
result = shap.attribution()

print("\n\n### OUTPUT ###")
print(result.output)

print("\n\n### ATTRIBUTION ###")
print(result.attribution)
```

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
| Sentences-/chunk-level attribution                                        | ✅                                                           | ❌ (token-only for text)        |
| Permanent context pinning (always-included features)                      | ✅                                                           | ❌                              |
| Pluggable similarity metric                                               | ✅ TF-IDF, embeddings                                        | ✅ TF-IDF, embeddings          |
| Docs & tutorial                                                           | ✅ Sphinx docs + tutorial                                    | ✅ README only                 |
| Unit tests & CI                                                           | ✅ Pytest + GitHub Actions                                   | ❌                              |
| Vision object attribution                                                 | —                                                            | ✅ PixelSHAP                   |
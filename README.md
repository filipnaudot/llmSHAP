<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="/docs/llmSHAP-logo-lightmode.png">
        <img alt="lighbench logo" src="/docs/llmSHAP-logo-darkmode.png" width="50%" height="50%">
    </picture>
</div>

# llmSHAP
![Unit Tests](https://github.com/filipnaudot/llmSHAP/actions/workflows/test.yml/badge.svg)

Lightweight interpretability framework using Shapley values for LLM-based outputs.

---

## Getting started

Install in editable mode with test dependencies:

```bash
pip install -e .[test]  # for bash
```
```bash
pip install -e '.[test]'  # for zsh
```

---

## Example usage

```python
from llmSHAP import DataHandler, BasicPromptHandler
from llmSHAP.llm import OpenAIInterface
from llmSHAP.attribution_methods import ShapleyAttribution

data = "In what city is the Eiffel Tower?"
handler = DataHandler(data, permanent_key_names={0,3,4})
prompt_handler = BasicPromptHandler(system="Answer the question briefly.")
llm = OpenAIInterface("gpt-4o-mini")

shap = ShapleyAttribution(model=llm, data_handler=handler, prompt_handler=prompt_handler, use_cache=True)
attribution, output = shap.attribution()

print("\n\n### OUPUT ###")
print(output)

print("\n\n### ATTRIBUTION ###")
print(attribution)
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

To exclude certain keys from the computations, use `permanent_key_names`:
```python
from llmSHAP import DataHandler

data = {"a": "The", "b": "quick", "c": "brown", "d": "fox"}
handler = DataHandler(data, permanent_key_names={"a", "d"})

print(handler.get_data(1, exclude_permanent_keys=True, mask=False))
# Output: {'b': 'quick'}

print(handler.get_data(1, exclude_permanent_keys=False, mask=False))
# Output: {'a': 'The', 'b': 'quick', 'd': 'fox'}
```
---
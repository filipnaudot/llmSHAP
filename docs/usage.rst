Usage
=====

Quick Start
-----------

Install (editable + tests):

.. code-block:: bash

   pip install -e .[test]     # bash
   pip install -e '.[test]'   # zsh

Minimal Example
---------------

.. code-block:: python

   from llmSHAP import DataHandler, BasicPromptCodec
   from llmSHAP.llm import OpenAIInterface
   from llmSHAP.attribution_methods import ShapleyAttribution

   data = "In what city is the Eiffel Tower?"
   handler = DataHandler(data, permanent_keys={0,3,4})
   prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
   llm = OpenAIInterface("gpt-4o-mini")

   shap = ShapleyAttribution(model=llm, data_handler=handler, prompt_codec=prompt_codec, use_cache=True)
   attribution, output = shap.attribution()

   print("### OUTPUT ###")
   print(output)
   print("### ATTRIBUTION ###")
   print(attribution)

Notes
-----

* ``OPENAI_API_KEY`` is read from your environment (``python-dotenv`` is supported).
* Autodoc mocks heavy/online imports for docs build, but at runtime you need real dependencies.
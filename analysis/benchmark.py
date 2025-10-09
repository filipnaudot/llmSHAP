# type: ignore
import csv

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
from llmSHAP.llm import OpenAIInterface
from llmSHAP.attribution_methods import CounterfactualSampler, SlidingWindowSampler, FullEnumerationSampler


def _load_data(file_name):
    data = []
    with open(file_name, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            disease = row["Disease"].strip()
            symptoms = [
                f"\nSYMPTOM: {value.strip().replace('_', ' ')}" for key, value in row.items()
                if key.startswith("Symptom_") and value and value.strip()
            ]
            data.append({"Disease": disease, "Symptoms": symptoms})
    return data


def _build_data_handler(data):
    prompt_dict = {"initial_query": "A patient is showing the following symptom(s):"}
    for i, symptom in enumerate(data["Symptoms"], start=1):
        prompt_dict[f"symptom_{i}"] = symptom
    prompt_dict["end_query"] = (
        "\nBased on these symptom(s), what disease or condition do you think they most likely have?"
        "\nANSWER BRIEFLY."
    )
    return DataHandler(prompt_dict, permanent_keys={"initial_query", "end_query"})





if __name__ == "__main__":
    VERBOSE = False

    prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
    llm = OpenAIInterface("gpt-4o-mini")
    data = _load_data("symptom_dataset.csv")
    
    attribution_results = {
        "CounterfactualSampler": [],
        "SlidingWindowSampler": [],
        "FullEnumerationSamplerCache": [],
        "FullEnumerationSampler": [],
    }
    for i, entry in enumerate(data):
        handler = _build_data_handler(entry)
        
        # Init all samplers
        players = handler.get_keys()
        samplers = [
            # name                          sampler                                  use_cache
            ("CounterfactualSampler",       CounterfactualSampler(),                 False),
            ("SlidingWindowSampler",        SlidingWindowSampler(players, w_size=3), False),
            ("FullEnumerationSamplerCache", FullEnumerationSampler(len(players)),    True),
            ("FullEnumerationSampler",      FullEnumerationSampler(len(players)),    False)
        ]

        # For each sampler, run attribution
        for name, sampler, cache in samplers:
            shap = ShapleyAttribution(model=llm,
                                    data_handler=handler,
                                    prompt_codec=prompt_codec,
                                    sampler=sampler,
                                    use_cache=cache,
                                    num_threads=7)
            result = shap.attribution()
            if VERBOSE:
                print("\n\n### OUTPUT ###")
                print(result.output)
                print("\n\n### ATTRIBUTION ###")
                print(result.attribution)

            # Store attribution
            attribution_results[name].append(result.attribution)


    print(attribution_results)
    # TODO: compare CounterfactualSampler and 
    # SlidingWindowSampler attributions to FullEnumerationSampler attribution
# type: ignore
import csv

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution
from llmSHAP.llm import OpenAIInterface



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
    prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
    llm = OpenAIInterface("gpt-4o-mini")
    data = _load_data("symptom_dataset.csv")
    
    for entry in data:
        handler = _build_data_handler(entry)
        shap = ShapleyAttribution(model=llm,
                                data_handler=handler,
                                prompt_codec=prompt_codec,
                                use_cache=True,
                                num_threads=7)
        result = shap.attribution()

        print("\n\n### OUTPUT ###")
        print(result.output)

        print("\n\n### ATTRIBUTION ###")
        print(result.attribution)

        print("\n\n### HEATMAP ###")
        print(result.render())
    
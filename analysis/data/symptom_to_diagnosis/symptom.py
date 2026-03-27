import csv
from pathlib import Path

from ..base import DataClass


class SymptomDataset(DataClass):
    SYSTEM_PROMPT = (
        "You are a clinical reasoning assistant. Given a patient's symptoms, "
        "answer with the single most likely disease or condition along with a motivation."
    )

    def __init__(self, disease_name: str, symptom_list: list[str]) -> None:
        self.disease_name = disease_name
        self.symptom_list = symptom_list

    def concepts(self) -> list[str]:
        return self.symptom_list

    @classmethod
    def system_prompt(cls) -> str:
        return cls.SYSTEM_PROMPT

    @classmethod
    def load(cls, file_path: str = "reduced_symptom_dataset.csv") -> list["SymptomDataset"]:
        csv_path = Path(file_path)
        if not csv_path.is_absolute() and not csv_path.exists():
            csv_path = Path(__file__).resolve().parent / csv_path
        data = []
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                disease_name = row["Disease"].strip()
                symptom_list = [
                    value.strip().replace("_", " ")
                    for key, value in row.items()
                    if key.startswith("Symptom_") and value and value.strip()
                ]
                data.append(cls(disease_name=disease_name, symptom_list=symptom_list))
        return data

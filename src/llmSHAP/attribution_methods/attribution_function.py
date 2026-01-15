import os, json
from dataclasses import asdict
import threading

from llmSHAP.types import ResultMapping, Optional
from llmSHAP.value_functions import ValueFunction
from llmSHAP.llm.llm_interface import LLMInterface

from llmSHAP.data_handler import DataHandler
from llmSHAP.prompt_codec import PromptCodec
from llmSHAP.generation import Generation
from llmSHAP.value_functions import TFIDFCosineSimilarity



class AttributionFunction:
    def __init__(self,
                 model: LLMInterface,
                 data_handler: DataHandler,
                 prompt_codec: PromptCodec,
                 use_cache: bool = False,
                 verbose: bool = True,
                 logging: bool = False,
                 log_filename: str = "log",
                 value_function: Optional[ValueFunction] = None,
                 ):
        self.model = model
        self.data_handler = data_handler
        self.prompt_codec = prompt_codec
        self.use_cache = use_cache
        self.verbose = verbose
        self.logging = logging
        self.log_filename = log_filename
        self.value_function = value_function or TFIDFCosineSimilarity()
        ####
        self.cache = {}
        self._cache_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.result: ResultMapping = {}

    def _v(self, base_generation: Generation, coalition_generation: Generation) -> float:
        return self.value_function(base_generation, coalition_generation)
    
    def _normalized_result(self) -> ResultMapping:
        total = sum([abs(value["score"]) for value in self.result.values()])
        if total == 0: return self.result
        return {key: {"value": value["value"], "score": value["score"] / total} for key, value in self.result.items()}
    
    def _get_output(self, coalition) -> Generation:
        frozen_coalition = frozenset(coalition)
        if self.use_cache:
            with self._cache_lock:
                if frozen_coalition in self.cache:
                    return self.cache[frozen_coalition]
        
        prompt = self.prompt_codec.build_prompt(self.data_handler, coalition)
        tools = self.prompt_codec.get_tools(self.data_handler, coalition)
        images = self.prompt_codec.get_images(self.data_handler, coalition)
        generation = self.model.generate(prompt, tools=tools, images=images)
        parsed_generation: Generation = self.prompt_codec.parse_generation(generation)
        
        if self.use_cache:
            with self._cache_lock:
                self.cache[frozen_coalition] = parsed_generation

        if self.logging:
            self._log(prompt, parsed_generation)
        
        return parsed_generation

    def _log(self, prompt, parsed_generation):
        os.makedirs("logs", exist_ok=True)
        log_data = {
                "prompt": prompt,
                "generation": asdict(parsed_generation)
            }

        log_path = os.path.join("logs", f"{self.log_filename}.jsonl")

        with self._log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)
                f.write("\n")

    def _add_feature_score(self, feature, score) -> None:
        for key, value in self.data_handler.get_data(feature, mask=False, exclude_permanent_keys=True).items():
            self.result[key] = {
                "value": value,
                "score": score
            }
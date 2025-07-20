from llmSHAP.types import ResultMapping
from llmSHAP.llm.llm_interface import LLMInterface

from llmSHAP.data_handler import DataHandler
from llmSHAP.prompt_handler import PromptHandler
from llmSHAP.generation import Generation
from llmSHAP.similarity_functions import EmbeddingCosineSimilarity



class AttributionFunction:
    def __init__(self,
                 model: LLMInterface,
                 data_handler: DataHandler,
                 prompt_handler: PromptHandler,
                 use_cache: bool = False,
                 verbose: bool = True):
        self.model = model
        self.data_handler = data_handler
        self.prompt_handler = prompt_handler
        self.use_cache = use_cache
        self.verbose = verbose
        ####
        self.cache = {}
        self.result: ResultMapping = {}
        self.similarity_function = EmbeddingCosineSimilarity()

    def _v(self, base_output: Generation, new_output: Generation) -> float:
        return self.similarity_function(str(base_output.output), str(new_output.output))
    
    def _normalized_result(self) -> ResultMapping:
        total = sum([abs(value["score"]) for value in self.result.values()])
        if total == 0: return self.result
        return {key: {"value": value["value"], "score": value["score"] / total} for key, value in self.result.items()}
    
    def _get_output(self, coalition, max_tokens: int = 512) -> Generation:
        frozen_coalition = frozenset(coalition)
        if self.use_cache and frozen_coalition in self.cache:
            return self.cache[frozen_coalition]
        
        prompt = self.prompt_handler.build_prompt(self.data_handler, coalition)
        generation = self.model.generate(prompt, max_tokens=max_tokens)
        parsed_generation: Generation = self.prompt_handler.parse_generation(generation)
        
        if self.use_cache:
            self.cache[frozen_coalition] = parsed_generation
        
        return parsed_generation

    def _add_feature_score(self, feature, score) -> None:
        for key, value in self.data_handler.get_data(feature, mask=False, exclude_permanent_keys=True).items():
            self.result[key] = {
                "value": value,
                "score": score
            }
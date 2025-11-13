import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import fsum

from llmSHAP.prompt_codec import PromptCodec
from llmSHAP.llm.llm_interface import LLMInterface
from llmSHAP.attribution_methods.attribution_function import AttributionFunction
from llmSHAP.attribution_methods.coalition_sampler import CoalitionSampler, FullEnumerationSampler
from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation
from llmSHAP.attribution import Attribution
from llmSHAP.value_functions import ValueFunction
from llmSHAP.types import Index, Optional


class ShapleyAttribution(AttributionFunction):
    def __init__(
        self,
        model: LLMInterface,
        data_handler: DataHandler,
        prompt_codec: PromptCodec,
        sampler: CoalitionSampler | None = None,
        use_cache: bool = False,
        verbose: bool = True,
        logging: bool = False,
        num_threads: int = 1,
        similarity_function: Optional[ValueFunction] = None,
    ):
        super().__init__(
            model,
            data_handler=data_handler,
            prompt_codec=prompt_codec,
            use_cache=use_cache,
            verbose=verbose,
            logging=logging,
            similarity_function=similarity_function,
        )
        self.num_threads = num_threads
        self.num_players = len(self.data_handler.get_keys(exclude_permanent_keys=True))
        self.sampler = sampler or FullEnumerationSampler(self.num_players)



    def _compute_marginal_contribution(self, coalition_set: set[Index], feature: Index, weight:float, base_generation: Generation):
        generation_without = self._get_output(coalition_set)
        generation_with = self._get_output(coalition_set | {feature})
        return weight * (self._v(base_generation, generation_with) - self._v(base_generation, generation_without))


    def attribution(self):
        start = time.perf_counter()
        base_generation: Generation = self._get_output(self.data_handler.get_keys())
        grand_coalition_value = self._v(base_generation, base_generation)
        empty_baseline_value = self._v(base_generation, self._get_output(set()))

        variable_keys = self.data_handler.get_keys(exclude_permanent_keys=True)

        with tqdm(self.data_handler.get_keys(), desc="Features", position=0, leave=False, disable=not self.verbose,) as feature_bar:
            for feature in feature_bar:
                if feature in self.data_handler.permanent_indexes: self._add_feature_score(feature, 0); continue

                shapley_value = 0.0
                tasks = []
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    for coalition_set, weight in list(self.sampler(feature, variable_keys)):
                        tasks.append(executor.submit(self._compute_marginal_contribution, coalition_set, feature, weight, base_generation,))

                    with tqdm(total=len(tasks), desc=f"Coalitions", position=1, leave=False, disable=not self.verbose) as coalition_bar:
                        contributions = []
                        for future in as_completed(tasks):
                            contributions.append(future.result())
                            coalition_bar.update(1)
                shapley_value = fsum(contributions)


                self._add_feature_score(feature, shapley_value)

        stop = time.perf_counter()
        if self.verbose: print(f"Time ({self.num_players} features): {(stop - start):.2f} seconds.")
        
        return Attribution(self.result, base_generation.output, empty_baseline_value, grand_coalition_value)
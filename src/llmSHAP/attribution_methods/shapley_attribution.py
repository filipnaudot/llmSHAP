import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from llmSHAP.prompt_codec import PromptCodec
from llmSHAP.llm.llm_interface import LLMInterface
from llmSHAP.attribution_methods.attribution_function import AttributionFunction
from llmSHAP.attribution_methods.coalition_sampler import CoalitionSampler, FullEnumerationSampler
from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation
from llmSHAP.attribution import Attribution
from llmSHAP.types import Index


class ShapleyAttribution(AttributionFunction):
    def __init__(
        self,
        model: LLMInterface,
        data_handler: DataHandler,
        prompt_codec: PromptCodec,
        sampler: CoalitionSampler | None = None,
        use_cache: bool = False,
        verbose: bool = True,
        logging:bool = False,
        num_threads: int = 1,
    ):
        super().__init__(
            model,
            data_handler=data_handler,
            prompt_codec=prompt_codec,
            use_cache=use_cache,
            verbose=verbose,
            logging=logging,
        )
        self.num_threads = num_threads
        self.num_players = len(self.data_handler.get_keys(exclude_permanent_keys=True))
        self.sampler = sampler or FullEnumerationSampler(self.num_players)



    def _compute_marginal_contribution(self, coalition_set: set[Index], feature: Index, weight:float, base_generation: Generation):
        if not coalition_set: return 0
        generation_without = self._get_output(coalition_set)
        generation_with = self._get_output(coalition_set | {feature})
        return weight * (self._v(base_generation, generation_with) - self._v(base_generation, generation_without))


    def attribution(self):
        start = time.perf_counter()
        base_generation: Generation = self._get_output(self.data_handler.get_keys())
        variable_keys = self.data_handler.get_keys(exclude_permanent_keys=True)

        with tqdm(self.data_handler.get_keys(), desc="Features", position=0, leave=False, disable=not self.verbose,) as feature_bar:
            for feature in feature_bar:
                if feature in self.data_handler.permanent_indexes: self._add_feature_score(feature, 0); continue

                shapley_value = 0.0
                tasks = []
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    for coalition_set, weight in self.sampler(feature, variable_keys):
                        tasks.append(executor.submit(self._compute_marginal_contribution, coalition_set, feature, weight, base_generation))

                for marginal_contribution in as_completed(tasks): shapley_value += marginal_contribution.result()

                self._add_feature_score(feature, shapley_value)

        stop = time.perf_counter()
        if self.verbose: print(f"Time ({self.num_players} features): {(stop - start):.2f} seconds.")
        return Attribution(self._normalized_result(), base_generation.output)
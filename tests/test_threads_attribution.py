import threading
import time

from llmSHAP.attribution_methods.shapley_attribution import ShapleyAttribution
from llmSHAP.llm.llm_interface import LLMInterface
from llmSHAP.generation import Generation
from llmSHAP.data_handler import DataHandler
from llmSHAP.prompt_codec import BasicPromptCodec
from llmSHAP.types import Optional, Any



class MockLLM(LLMInterface):
    def generate(self, prompt, tools: Optional[list[Any]], images: Optional[list[Any]]) -> str:
        return str(prompt)

class CountingMockLLM(LLMInterface):
    def __init__(self):
        self._lock = threading.Lock()
        self.call_count = 0

    def generate(self, prompt, tools: Optional[list[Any]], images: Optional[list[Any]]) -> str:
        with self._lock:
            self.call_count += 1
        time.sleep(0.01)
        return str(prompt)


class ShapleyLenV(ShapleyAttribution):
    def _v(self, base_output: Generation, new_output: Generation) -> float:
        return float(len(str(new_output.output)))


def test_attribution_same_with_single_and_multi_threads():
    data = "Lorem ipsum dolor sit amet"
    data_handler = DataHandler(data)
    prompt_codec = BasicPromptCodec()
    llm = MockLLM()

    single_thread = ShapleyLenV(model=llm,
                                data_handler=data_handler,
                                prompt_codec=prompt_codec,
                                use_cache=True,
                                verbose=False,
                                logging=False,
                                num_threads=1)
    single_thread.attribution()
    single_thread_res = single_thread.result


    multi_thread = ShapleyLenV(model=llm,
                               data_handler=data_handler,
                               prompt_codec=prompt_codec,
                               use_cache=True,
                               verbose=False,
                               logging=False,
                               num_threads=4)
    multi_thread.attribution()
    multi_thread_res = multi_thread.result
    assert single_thread_res == multi_thread_res


def test_cache_deduplicates_inflight_coalitions_across_threads():
    data = "one two three"
    data_handler = DataHandler(data)
    prompt_codec = BasicPromptCodec()
    llm = CountingMockLLM()

    shap = ShapleyLenV(model=llm,
                       data_handler=data_handler,
                       prompt_codec=prompt_codec,
                       use_cache=True,
                       verbose=False,
                       logging=False,
                       num_threads=4)
    shap.attribution()
    assert llm.call_count == 2 ** len(data_handler.get_keys())

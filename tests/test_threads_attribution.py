from llmSHAP.attribution_methods.shapley_attribution import ShapleyAttribution
from llmSHAP.llm.llm_interface import LLMInterface
from llmSHAP.generation import Generation
from llmSHAP.data_handler import DataHandler
from llmSHAP.prompt_codec import BasicPromptCodec
from llmSHAP.types import Optional, Any

class MockLLM(LLMInterface):
    def generate(self, prompt, tools: Optional[list[Any]], images: Optional[list[Any]]) -> str:
        return str(prompt)
    def name(self): return "fake"
    def is_local(self): return True
    def cleanup(self): pass


class ShapleyLenV(ShapleyAttribution):
    def _v(self, base_output: Generation, new_output: Generation) -> float:
        return float(len(str(new_output.output)))


def test_attribution_same_with_single_and_multi_threads():
    data = "Lorem ipsum dolor sit amet"
    data_handler = DataHandler(data)
    prompt_codec = BasicPromptCodec()
    llm = MockLLM()

    single_thread = ShapleyLenV(
        model=llm,
        data_handler=data_handler,
        prompt_codec=prompt_codec,
        use_cache=True,
        verbose=False,
        logging=False,
        num_threads=1,
    )
    single_thread.attribution()
    single_thread_res = single_thread.result


    multi_thread = ShapleyLenV(
        model=llm,
        data_handler=data_handler,
        prompt_codec=prompt_codec,
        use_cache=True,
        verbose=False,
        logging=False,
        num_threads=4,
    )
    multi_thread.attribution()
    multi_thread_res = multi_thread.result

    assert single_thread_res == multi_thread_res
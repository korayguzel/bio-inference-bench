# bio-inference-bench: Profiling benchmark for autoregressive protein sequence generation

from bio_inference_bench.int8_generate import generate
from bio_inference_bench.models import load_model_and_tokenizer

__all__ = ["generate", "load_model_and_tokenizer"]

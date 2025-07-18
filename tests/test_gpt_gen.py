import pytest
import tiktoken

from models import GPTModel
from utils import generate_text, ids2text, text2ids


class TestGPTGen:
    @classmethod
    def setup_class(cls):
        cls.model = GPTModel(vocab_size=50257, emb_dim=32, context_len=128, n_heads=8, n_layers=4)
        cls.tokenizer = tiktoken.get_encoding("gpt2")
        cls.context_size = 128

    @pytest.mark.parametrize("num_tokens", [5, 10, 20, 1024])
    def test_generate_text(self, num_tokens):
        text = "Hello, world!"
        ids = text2ids(text, self.tokenizer)
        generated_ids = generate_text(self.model, ids, num_tokens, self.context_size, use_cache=True)
        generated_text = ids2text(generated_ids, self.tokenizer)
        assert generated_text.startswith("Hello, world!")
        print(generated_text)

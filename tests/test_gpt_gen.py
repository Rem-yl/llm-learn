import pytest
import tiktoken

from models import GPTModel


class TestGPTGen:
    @classmethod
    def setup_class(cls):
        cls.model = GPTModel(vocab_size=50257, emb_dim=32, context_len=128, n_heads=8, n_layers=4)
        cls.tokenizer = tiktoken.get_encoding("gpt2")
        cls.context_size = 128

    @pytest.mark.parametrize("num_tokens", [5, 10, 20, 1024])
    def test_generate_text(self, num_tokens):
        text = "Hello, world!"
        generated_text = self.model.gen_text(text, max_tokens=num_tokens, context_size=self.context_size)
        assert generated_text.startswith("Hello, world!")
        print(generated_text)

from src.tokenizer import tokenize, pre_token_score

def test_tokenize_simple():
    assert tokenize("Hello world!") == ["hello", "world"]

def test_pre_token_score_default():
    assert pre_token_score({}) == 1.0

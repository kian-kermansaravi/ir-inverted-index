import re

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens
def pre_token_score(token: str) -> float:
    return 1.0

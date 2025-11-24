from src.tokenizer import tokenize
from src.inverted_index import InvertedIndex

if __name__ == "__main__":
    text = "Hello World, this is a test."
    tokens = tokenize(text)
    print("TOKENS:", tokens)

    index = InvertedIndex()
    index.add_document(1, tokens)
    print("INVERTED INDEX:", index.index)

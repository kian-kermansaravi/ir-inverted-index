from src.inverted_index import InvertedIndex

def test_index_and_postings():
    idx = InvertedIndex()
    idx.index_document("d1", "hello world hello", {"length": 3})
    p = idx.get_postings("hello")
    assert any(x["doc"] == "d1" for x in p)
    score = idx.compute_tf_idf("hello", "d1")
    assert score > 0

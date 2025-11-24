from src.btree import BTree

def test_btree_insert_search():
    t = BTree(t=2)
    t.insert("apple", {"df":1})
    t.insert("banana", {"df":1})
    assert t.search(t.root, "apple") is not None
    assert t.search(t.root, "banana") is not None

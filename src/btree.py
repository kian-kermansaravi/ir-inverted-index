from typing import List, Optional

class BTreeNode:
    def __init__(self, t: int, leaf: bool = True):
        self.t = t
        self.keys: List[str] = []
        self.values: List = []
        self.children: List["BTreeNode"] = []
        self.leaf = leaf

    def is_full(self):
        return len(self.keys) >= 2 * self.t - 1

class BTree:
    def __init__(self, t: int = 2):
        self.t = t
        self.root = BTreeNode(t, leaf=True)

    def search(self, node: Optional[BTreeNode], key: str):
        if node is None:
            return None
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == key:
            return node.values[i]
        if node.leaf:
            return None
        return self.search(node.children[i], key)

    def split_child(self, parent: BTreeNode, index: int):
        t = self.t
        node = parent.children[index]
        new_node = BTreeNode(t, leaf=node.leaf)
        # middle key moves up
        mid_key = node.keys[t-1]
        mid_val = node.values[t-1] if node.values else None

        # split keys/values
        new_node.keys = node.keys[t:]
        new_node.values = node.values[t:]
        node.keys = node.keys[:t-1]
        node.values = node.values[:t-1]

        if not node.leaf:
            new_node.children = node.children[t:]
            node.children = node.children[:t]

        parent.children.insert(index+1, new_node)
        parent.keys.insert(index, mid_key)
        parent.values.insert(index, mid_val)

    def insert_non_full(self, node: BTreeNode, key: str, value):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            node.values.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i+1] = node.keys[i]
                node.values[i+1] = node.values[i]
                i -= 1
            node.keys[i+1] = key
            node.values[i+1] = value
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if node.children[i].is_full():
                self.split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self.insert_non_full(node.children[i], key, value)

    def insert(self, key: str, value):
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(root)
            self.root = new_root
            self.split_child(new_root, 0)
            self.insert_non_full(new_root, key, value)
        else:
            self.insert_non_full(root, key, value)

    def traverse(self, node=None, depth=0):
        if node is None:
            node = self.root
        s = "  " * depth + f"Keys: {node.keys}\n"
        if not node.leaf:
            for c in node.children:
                s += self.traverse(c, depth+1)
        return s

"""Minimal B-tree implementation for storing dictionary terms."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Optional, Tuple


@dataclass
class BTreeNode:
    min_degree: int
    leaf: bool = True

    def __post_init__(self) -> None:
        self.keys: List[str] = []
        self.values: List[Any] = []
        self.children: List[BTreeNode] = []

    @property
    def is_full(self) -> bool:
        return len(self.keys) == 2 * self.min_degree - 1


class BTree:
    def __init__(self, min_degree: int = 3) -> None:
        if min_degree < 2:
            raise ValueError("min_degree must be at least 2")
        self.min_degree = min_degree
        self.root = BTreeNode(min_degree=min_degree)

    def search(self, key: str) -> Optional[Any]:
        node = self.root
        while True:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and key == node.keys[i]:
                return node.values[i]
            if node.leaf:
                return None
            node = node.children[i]

    def insert(self, key: str, value: Any, merge_fn: Optional[Callable[[Any, Any], Any]] = None) -> None:
        # Fast path: if key already exists anywhere, merge/update in place.
        existing = self.search(key)
        if existing is not None:
            if merge_fn:
                merge_fn(existing, value)
            else:
                self._replace_value(self.root, key, value)
            return

        root = self.root
        if root.is_full:
            new_root = BTreeNode(min_degree=self.min_degree, leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, key, value, merge_fn)
        else:
            self._insert_non_full(root, key, value, merge_fn)

    def _replace_value(self, node: BTreeNode, key: str, value: Any) -> bool:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            node.values[i] = value
            return True
        if node.leaf:
            return False
        return self._replace_value(node.children[i], key, value)

    def _insert_non_full(
        self, node: BTreeNode, key: str, value: Any, merge_fn: Optional[Callable[[Any, Any], Any]]
    ) -> None:
        i = len(node.keys) - 1
        if node.leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            if i >= 0 and key == node.keys[i]:
                node.values[i] = merge_fn(node.values[i], value) if merge_fn else value
                return
            node.keys.insert(i + 1, key)
            node.values.insert(i + 1, value)
            return

        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        if i < len(node.keys) and key == node.keys[i]:
            node.values[i] = merge_fn(node.values[i], value) if merge_fn else value
            return

        if node.children[i].is_full:
            self._split_child(node, i)
            if key > node.keys[i]:
                i += 1
            elif key == node.keys[i]:
                node.values[i] = merge_fn(node.values[i], value) if merge_fn else value
                return
        self._insert_non_full(node.children[i], key, value, merge_fn)

    def _split_child(self, parent: BTreeNode, index: int) -> None:
        full_child = parent.children[index]
        t = self.min_degree
        new_node = BTreeNode(min_degree=t, leaf=full_child.leaf)

        new_node.keys = full_child.keys[t:]
        new_node.values = full_child.values[t:]

        mid_key = full_child.keys[t - 1]
        mid_val = full_child.values[t - 1]
        full_child.keys = full_child.keys[: t - 1]
        full_child.values = full_child.values[: t - 1]

        if not full_child.leaf:
            new_node.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

        parent.keys.insert(index, mid_key)
        parent.values.insert(index, mid_val)
        parent.children.insert(index + 1, new_node)

    def level_strings(self) -> List[str]:
        if not self.root.keys:
            return []
        lines: List[str] = []
        queue: Deque[Tuple[BTreeNode, int]] = deque([(self.root, 0)])
        current_level = 0
        level_nodes: List[str] = []

        while queue:
            node, level = queue.popleft()
            if level != current_level:
                lines.append("  ".join(level_nodes))
                level_nodes = []
                current_level = level
            level_nodes.append("[" + " ".join(node.keys) + "]")
            for child in node.children:
                queue.append((child, level + 1))

        if level_nodes:
            lines.append("  ".join(level_nodes))
        return lines

    def pretty_print(self) -> str:
        return "\n".join(self.level_strings()) or "<empty>"

    def __len__(self) -> int:
        return self._count_nodes(self.root)

    def _count_nodes(self, node: BTreeNode) -> int:
        total = 1
        for child in node.children:
            total += self._count_nodes(child)
        return total

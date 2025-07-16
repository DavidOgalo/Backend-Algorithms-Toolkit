"""
Binary Search Tree (BST) Implementation for Ordered Data Access

Why Use BSTs in Backend Development?
--------------------------------------------------
Binary Search Trees are essential when you need:
- **Ordered Data Access**: BSTs maintain sorted order, unlike hash tables
- **Range Queries**: Efficiently find all records between two values
- **Time-Series Data**: Query data by timestamp ranges
- **Database Indexing**: Many database systems use B-trees (generalized BSTs)

This implementation is designed for backend engineers who need fast, reliable, and maintainable ordered data structures.
"""
from typing import Optional, Any, List, Tuple

class TreeNode:
    """
    Node for the Binary Search Tree.
    Stores a key-value pair and pointers to left/right children.
    """
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.height = 1  # For AVL balancing (not implemented here)

class BinarySearchTree:
    """
    Binary Search Tree for maintaining sorted data with O(log n) average-case operations.

    Supports:
    - Insertion
    - Search
    - In-order traversal (sorted order)
    - Range queries
    """
    def __init__(self):
        self.root: Optional[TreeNode] = None
        self.size = 0

    def insert(self, key: str, value: Any) -> None:
        """
        Insert a key-value pair into the BST, maintaining the BST property.
        If the key already exists, its value is updated.
        """
        self.root, inserted_new = self._insert_recursive(self.root, key, value)
        if inserted_new:
            self.size += 1

    def _insert_recursive(self, node: Optional[TreeNode], key: str, value: Any) -> Tuple[Optional[TreeNode], bool]:
        """
        Helper for insert. Returns (node, inserted_new: bool)
        """
        if node is None:
            return TreeNode(key, value), True

        if key < node.key:
            node.left, inserted_new = self._insert_recursive(node.left, key, value)
        elif key > node.key:
            node.right, inserted_new = self._insert_recursive(node.right, key, value)
        else:
            # Key exists, update value
            node.value = value
            inserted_new = False

        # Update height for potential balancing (not used here)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        return node, inserted_new

    def _get_height(self, node: Optional[TreeNode]) -> int:
        """Return the height of a node (for AVL balancing, not used here)."""
        return node.height if node else 0

    def search(self, key: str) -> Optional[Any]:
        """
        Search for a key and return its value if found, else None.
        """
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node: Optional[TreeNode], key: str) -> Optional[Any]:
        if node is None:
            return None
        if key == node.key:
            return node.value
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    def inorder_traversal(self) -> List[Tuple[str, Any]]:
        """
        Return all key-value pairs in sorted order as a list of (key, value) tuples.
        """
        result: List[Tuple[str, Any]] = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node: Optional[TreeNode], result: List[Tuple[str, Any]]):
        if node:
            self._inorder_recursive(node.left, result)
            result.append((node.key, node.value))
            self._inorder_recursive(node.right, result)

    def range_query(self, start_key: str, end_key: str) -> List[Tuple[str, Any]]:
        """
        Find all key-value pairs in the range [start_key, end_key] (inclusive).
        Efficient for time-based queries, price ranges, or date filtering.
        """
        result: List[Tuple[str, Any]] = []
        self._range_query_recursive(self.root, start_key, end_key, result)
        return result

    def _range_query_recursive(self, node: Optional[TreeNode], start: str, end: str, result: List[Tuple[str, Any]]):
        if node is None:
            return
        # Search left subtree if possible
        if node.key > start:
            self._range_query_recursive(node.left, start, end, result)
        # Add key if in range
        if start <= node.key <= end:
            result.append((node.key, node.value))
        # Search right subtree if possible
        if node.key < end:
            self._range_query_recursive(node.right, start, end, result)


# Example: Using BST for time-series data indexing
if __name__ == "__main__":
    time_series_index = BinarySearchTree()
    time_series_index.insert("2024-06-04T10:00:00", {"temperature": 25.5, "humidity": 60})
    time_series_index.insert("2024-06-04T11:00:00", {"temperature": 26.2, "humidity": 58})
    time_series_index.insert("2024-06-04T12:00:00", {"temperature": 27.1, "humidity": 55})

    # Range query for data between specific times
    morning_data = time_series_index.range_query("2024-06-04T10:00:00", "2024-06-04T11:30:00")
    print(f"Morning data: {morning_data}")
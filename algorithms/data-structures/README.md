
# Data Structures: Binary Search Tree & Hash Table

Reusable, efficient, and production-ready data structures for backend and algorithmic use. Both the Binary Search Tree (BST) and Hash Table are implemented in Python with clear APIs and can be imported as modules in other algorithms or backend systems.

---

## ✨ Features

- **Binary Search Tree (BST):**
  - Maintains sorted order of keys
  - Fast O(log n) average-case insert, search, and range queries
  - Supports in-order traversal and efficient range queries (e.g., time-series, price ranges)
  - Well-documented and extensible for backend indexing and analytics

- **Hash Table:**
  - O(1) average-case insert, lookup, and delete
  - Dynamic resizing and collision handling
  - Ideal for caching, session management, and fast key-value storage
  - Simple, Pythonic API for backend integration

---

## 🧩 Use Cases

- Fast lookups, inserts, and updates for backend data
- Caching and session management (Hash Table)
- Time-series and range queries (BST)
- Underlying modules for other algorithms (e.g., rate limiting, analytics, graph algorithms)
- Database indexing, search, and deduplication

---

## 📦 Importing and Using as Modules

You can import these data structures into other algorithms or backend services as needed:

```python
# Import BST and HashTable from the data-structures package
from data_structures.binary_search_tree import BinarySearchTree
from data_structures.hash_table import HashTable

# Example: Use BST for time-series data
bst = BinarySearchTree()
bst.insert("2025-07-16T10:00:00", {"temperature": 25.5})
result = bst.range_query("2025-07-16T09:00:00", "2025-07-16T11:00:00")
print(result)

# Example: Use HashTable for caching
cache = HashTable()
cache.put("user_123", {"session": "abc", "expires": "2025-12-31"})
print(cache.get("user_123"))
```

---

## 📄 Implementation & Examples

See [`binary_search_tree.py`](./binary_search_tree.py) and [`hash_table.py`](./hash_table.py) for full implementations, detailed docstrings, and more usage examples.

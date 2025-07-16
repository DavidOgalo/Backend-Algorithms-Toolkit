"""
Hash Table Implementation for Fast Lookups and Caching

Hash tables are the backbone of many backend systems because they provide O(1) average-case time complexity for insertions, deletions, and lookups. 

Common backend use cases:
- **Session Management**: Storing user sessions with instant lookup
- **Caching Systems**: Redis and Memcached use hash table principles
- **Database Indexing**: Most database indexes use hash-based structures
- **API Rate Limiting**: Quick lookups for request counts per user
"""

from typing import Any, Optional

class HashTable:
    """
    Custom hash table for caching and fast lookups.

    Features:
    - Dynamic resizing (rehashing)
    - Collision handling via chaining
    - O(1) average-case insert, get, and delete
    """
    def __init__(self, initial_size: int = 16):
        """
        Initialize the hash table.
        :param initial_size: Number of buckets to start with (default: 16)
        """
        self.size = initial_size
        self.count = 0
        self.buckets = [[] for _ in range(self.size)]
        self.load_factor_threshold = 0.75

    def _hash(self, key: str) -> int:
        """
        Compute a hash for the given key using a simple polynomial rolling hash.
        :param key: The key to hash
        :return: Hash index for the key
        """
        hash_value = 0
        for char in key:
            hash_value = (hash_value * 31 + ord(char)) % self.size
        return hash_value

    def _resize(self):
        """
        Resize the hash table when the load factor threshold is exceeded.
        Rehashes all existing key-value pairs into the new buckets.
        """
        old_buckets = self.buckets
        self.size *= 2
        self.count = 0
        self.buckets = [[] for _ in range(self.size)]
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key: str, value: Any) -> None:
        """
        Insert or update a key-value pair in the hash table.
        :param key: The key to insert or update
        :param value: The value to associate with the key
        """
        if self.count >= self.size * self.load_factor_threshold:
            self._resize()

        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]

        # Check if key already exists and update
        for index, (k, _) in enumerate(bucket):
            if k == key:
                bucket[index] = (key, value)
                return

        # Add a new key-value pair
        bucket.append((key, value))
        self.count += 1

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with a key, or None if not found.
        :param key: The key to look up
        :return: The value if found, else None
        """
        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]
        for k, v in bucket:
            if k == key:
                return v
        return None

    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair from the hash table.
        :param key: The key to delete
        :return: True if the key was found and deleted, False otherwise
        """
        hash_index = self._hash(key)
        bucket = self.buckets[hash_index]
        for index, (k, _) in enumerate(bucket):
            if k == key:
                del bucket[index]
                self.count -= 1
                return True
        return False


# Example usage in backend caching system
if __name__ == "__main__":
    session_cache = HashTable()
    session_cache.put("JohnDoe_session", {"user_id": 254, "expires": "2025-12-31"})
    print(session_cache.get("JohnDoe_session"))
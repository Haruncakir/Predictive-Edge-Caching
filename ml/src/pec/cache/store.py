"""CacheStore — byte-addressed, fixed-capacity content store.

This is the "Cache Store" box inside the Edge Cache Controller block.
It holds ``(file_id, size_bytes)`` pairs and enforces a total byte budget.

DESIGN CHOICES
──────────────
* **Byte-based capacity** is the default and only mode.  Real edge caches
  are byte-limited (Cherkasova, "Improving WWW proxies performance with
  Greedy-Dual-Size-Frequency caching policy", HP Labs, 1998).

* Internally backed by a ``dict[file_id, size_bytes]`` — O(1) lookup, O(N)
  iteration for victim selection (N = number of cached files, typically
  small relative to the catalog).

* No locking: the cache controller is single-threaded in the Python
  simulation; concurrency lives on the C++ producer side.
"""

from __future__ import annotations


class CacheStore:
    """Fixed-capacity byte-addressed cache store.

    Parameters
    ----------
    capacity_bytes : int
        Maximum total payload bytes the cache can hold.
    """

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")
        self._capacity = capacity_bytes
        self._store: dict[int, int] = {}   # file_id → size_bytes
        self._used: int = 0

    # ── Queries ───────────────────────────────────────────────────────

    def contains(self, file_id: int) -> bool:
        return file_id in self._store

    @property
    def file_ids(self) -> set[int]:
        return set(self._store.keys())

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return len(self._store)

    @property
    def utilisation(self) -> float:
        """Fraction of capacity currently used, in [0, 1]."""
        return self._used / self._capacity if self._capacity else 0.0

    def free_bytes(self) -> int:
        return self._capacity - self._used

    def file_size(self, file_id: int) -> int:
        """Return the stored size of a cached file, or 0 if absent."""
        return self._store.get(file_id, 0)

    # ── Mutations ─────────────────────────────────────────────────────

    def admit(self, file_id: int, size_bytes: int) -> bool:
        """Try to admit a file.  Returns True on success.

        Fails (returns False) if the file is larger than remaining free
        space.  Callers must evict first to make room.  A file that is
        already cached is a silent no-op (returns True).
        """
        if file_id in self._store:
            return True  # already cached
        if size_bytes > self.free_bytes():
            return False
        self._store[file_id] = size_bytes
        self._used += size_bytes
        return True

    def evict(self, file_id: int) -> int:
        """Remove a file from the cache.  Returns freed bytes (0 if absent)."""
        size = self._store.pop(file_id, 0)
        self._used -= size
        return size

    def clear(self) -> None:
        self._store.clear()
        self._used = 0

    def __repr__(self) -> str:
        pct = self.utilisation * 100
        return (
            f"CacheStore(used={self._used}/{self._capacity} "
            f"[{pct:.1f}%], files={self.count})"
        )

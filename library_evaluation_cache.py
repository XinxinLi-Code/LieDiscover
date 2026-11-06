"""Bounded reuse of immutable numerical library evaluations.

The cache never stores rewards or episode state. Entries are private, read-only
tensor snapshots: callers may replace their row list, but must not edit tensors
in place. Packing only copies entries; it performs no numerical reductions.
"""

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
import sys

import torch


class PackedRows(list):
    """List-compatible matrix rows backed by one contiguous sample tensor."""

    def __init__(self, tensor):
        if tensor.ndim != 3:
            raise ValueError("Packed constraint rows require [sample, equation, column].")
        self.packed = tensor
        super().__init__(tensor.unbind(0))

    def copy(self):
        result = PackedRows.__new__(PackedRows)
        result.packed = self.packed
        list.__init__(result, self)
        return result


def pack_rows(rows):
    if isinstance(rows, PackedRows):
        return rows
    if not rows or any(row.shape != rows[0].shape for row in rows):
        return rows
    return PackedRows(torch.stack(tuple(rows), dim=0))


@dataclass(frozen=True)
class LibraryEvaluation:
    rows: list
    scales: torch.Tensor
    gram: torch.Tensor
    decision: object


def _entry_bytes(entry):
    """Conservative retained storage plus Python-object accounting.

    Tensor views count their backing storage once, including portions outside
    the view. Shared storage across entries is deliberately counted again.
    """
    objects, storages = set(), set()

    def visit(value):
        if id(value) in objects:
            return 0
        objects.add(id(value))
        size = sys.getsizeof(value)
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            # Include a conservative allowance for TensorImpl/view metadata.
            size += 256
            if key not in storages:
                storages.add(key)
                size += storage.nbytes()
        elif is_dataclass(value):
            size += sum(visit(getattr(value, field.name)) for field in fields(value))
        elif isinstance(value, dict):
            size += sum(visit(key) + visit(item) for key, item in value.items())
        elif isinstance(value, (tuple, list)):
            size += sum(visit(item) for item in value)
            if isinstance(value, PackedRows):
                size += visit(value.packed)
        return size

    return visit(entry)


class LibraryEvaluationCache:
    """Environment-local LRU, invalidated with the fixed sample context."""

    def __init__(self, max_bytes=128 * 1024**2, max_entries=256):
        if int(max_bytes) < 0 or int(max_entries) < 0:
            raise ValueError("Numerical cache limits must be nonnegative.")
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        self._entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.peak_bytes = self.oversize_skips = 0

    @property
    def enabled(self):
        return self.max_bytes > 0 and self.max_entries > 0

    def __len__(self):
        return len(self._entries)

    def statistics(self):
        """Return host-only counters; never touch numerical cache contents."""
        lookups = self.hits + self.misses
        return {
            "entries": len(self), "max_entries": self.max_entries,
            "bytes": self.bytes, "max_bytes": self.max_bytes,
            "peak_bytes": self.peak_bytes,
            "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / lookups if lookups else 0.0,
            "evictions": self.evictions, "oversize_skips": self.oversize_skips,
        }

    def clear(self):
        self._entries.clear()
        self.bytes = 0

    def get(self, key, scales):
        item = self._entries.get(key) if self.enabled else None
        if item is not None and torch.equal(item[0].scales, scales):
            self._entries.move_to_end(key)
            self.hits += 1
            return item[0]
        self.misses += 1
        return None

    def put(self, key, entry):
        if not self.enabled:
            return
        # Isolate the mutable row container; numerical tensor storage is shared
        # read-only, just as for the existing per-term feature cache.
        entry = LibraryEvaluation(entry.rows.copy(), entry.scales, entry.gram, entry.decision)
        size = _entry_bytes(entry) + sys.getsizeof(key)
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if size > self.max_bytes:
            self.oversize_skips += 1
            return
        while self._entries and (self.bytes + size > self.max_bytes
                                 or len(self._entries) >= self.max_entries):
            _, (_, removed) = self._entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self._entries[key] = (entry, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

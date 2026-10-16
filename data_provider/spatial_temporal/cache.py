"""运行内有界 transformed-sample 缓存；序列化隔离调用方写入，不读外部 pickle。"""
from collections import OrderedDict
import pickle

from .schema import canonical_hash


class SampleCache:
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, dataset, transform, index):
        key = (dataset.fingerprint, canonical_hash(dataset.config), transform.fingerprint, dataset.split, index)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return pickle.loads(self.entries[key])
        self.misses += 1
        sample = transform.apply(dataset[index])
        if not self.budget_bytes:
            return sample
        payload = pickle.dumps(sample, protocol=5)
        if len(payload) <= self.budget_bytes:
            while self.bytes + len(payload) > self.budget_bytes:
                _, expired = self.entries.popitem(last=False)
                self.bytes -= len(expired)
                self.evictions += 1
            self.entries[key] = payload
            self.bytes += len(payload)
        return sample

    def report(self):
        return {"budget_bytes": self.budget_bytes, "retained_payload_bytes": self.bytes,
                "entries": len(self.entries), "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "scope": "serialized sample payload; excludes Python index and live batch"}

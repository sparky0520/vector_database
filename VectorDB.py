import json
import numpy as np


class VectorDB:
    def __init__(self):
        self.vectors = {}  # id -> vector
        self.metadata = {}  # id => metadata

    def add(self, vector_id: str, vector, metadata=None):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")

        self.vectors[vector_id] = vector
        self.metadata[vector_id] = metadata or {}

    def cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b)) / (norm_a * norm_b)

    def search(self, query_vector, top_k=5):
        query_vector = np.asarray(query_vector, dtype=np.float32)

        results = []

        for vector_id, vector in self.vectors.items():
            score = self.cosine_similarity(query_vector, vector)

            results.append(
                {"id": vector_id, "score": score, "metadata": self.metadata[vector_id]}
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path):
        data = {
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.vectors = {
            k: np.array(v, dtype=np.float32) for k, v in data["vectors"].items()
        }

        self.metadata = data["metadata"]

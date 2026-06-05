import numpy as np


class MatrixVectorDB:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = None
        self.ids = []

    def add_bulk(self, vectors):

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Expected shape (N,D)")

        self.vectors = vectors

        self.ids = np.arange(len(vectors))

        # normalize once
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)

        self.vectors = self.vectors / np.maximum(norms, 1e-12)

    def search(self, query, top_k=10):

        query = np.asarray(query, dtype=np.float32)

        query = query / np.maximum(np.linalg.norm(query), 1e-12)

        scores = self.vectors @ query

        top_indices = np.argpartition(scores, -top_k)[-top_k:]

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

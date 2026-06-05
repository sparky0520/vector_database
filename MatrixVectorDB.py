import numpy as np
import time


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


# --------------------------
# Benchmark
# --------------------------

DIMENSION = 384


def benchmark(size):

    print()
    print("=" * 60)
    print(f"{size:,} vectors")
    print("=" * 60)

    vectors = np.random.rand(size, DIMENSION).astype(np.float32)

    db = MatrixVectorDB(DIMENSION)

    start = time.perf_counter()

    db.add_bulk(vectors)

    build_time = time.perf_counter() - start

    query = np.random.rand(DIMENSION).astype(np.float32)

    start = time.perf_counter()

    result = db.search(query, top_k=10)

    search_time = time.perf_counter() - start

    print(f"Build Time : {build_time:.3f}s")

    print(f"Search Time: {search_time:.6f}s")

    print(f"Top Score  : {result[0][1]:.4f}")

    return search_time


if __name__ == "__main__":
    sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000]

    summary = []

    for size in sizes:
        t = benchmark(size)

        summary.append((size, t))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for size, t in summary:
        print(f"{size:>10,} vectors -> {t:.6f}s")

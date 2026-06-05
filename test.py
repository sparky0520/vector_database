import gc
import time
import numpy as np

from main import VectorDB

# ---------------------------
# Benchmark Helpers
# ---------------------------

DIMENSION = 384


def generate_vectors(count):
    return np.random.rand(count, DIMENSION).astype(np.float32)


def benchmark(size):
    print("=" * 60)
    print(f"Testing {size:,} vectors")
    print("=" * 60)

    db = VectorDB()

    print("Generating vectors...")
    vectors = generate_vectors(size)

    print("Loading into database...")

    start_load = time.perf_counter()

    for i, vec in enumerate(vectors):
        db.add(f"vec_{i}", vec)

    load_time = time.perf_counter() - start_load

    print(f"Load Time: {load_time:.2f}s")

    query = np.random.rand(DIMENSION).astype(np.float32)

    print("Running search...")

    start_search = time.perf_counter()

    results = db.search(query, top_k=10)

    search_time = time.perf_counter() - start_search

    print(f"Search Time: {search_time:.4f}s")
    print()

    print("Top Result:")
    print(results[0])

    del db
    del vectors
    gc.collect()

    return search_time


if __name__ == "__main__":
    sizes = [
        1_000,
        10_000,
        100_000,
        500_000,
        1_000_000,
    ]

    summary = []

    for size in sizes:
        try:
            t = benchmark(size)

            summary.append((size, t))

        except MemoryError:
            print(f"MemoryError at {size:,} vectors")
            break

    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for size, t in summary:
        print(f"{size:>10,} vectors -> {t:.4f}s")

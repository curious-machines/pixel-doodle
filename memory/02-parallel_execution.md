---
name: Parallel execution architecture
description: Tile-based parallelism model — buffer partitioning, read-only sharing, false sharing avoidance
type: project
---

The execution model splits the pixel buffer into tiles and distributes tiles to worker threads. Each thread calls a JIT'd function on its tile. Within a tile, SIMD processes multiple pixels per instruction.

Architecture:
```
thread pool splits buffer → N worker threads
  each thread calls JIT'd function on its tile
    JIT'd function uses SIMD within its tile
```

Key rules:
- Source/input data is shared read-only across all threads (no synchronization needed)
- Output buffer is partitioned so each thread writes exclusively to its own region
- Double-buffering (separate input and output images) is needed if a kernel reads from the output
- Tile sizes should align to cache line boundaries (64 bytes) to avoid false sharing
- Work-stealing thread pools (rayon, oneTBB) are preferred for fine-grained tasks; a simple shared queue works for coarse-grained tiles

**Why:** This architecture avoids locks in the hot path entirely. Read-only sharing is free, and exclusive write regions eliminate contention.

**How to apply:** When implementing any pixel-processing kernel, ensure the input/output separation is maintained. When choosing tile sizes, consider both SIMD width alignment and cache line alignment.

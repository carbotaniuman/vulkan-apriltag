module union_find;

public uint32_t uf_load(Atomic<uint32_t> *uf, uint32_t n) {
    return uf[n].load(MemoryOrder.Relaxed);
}

public uint32_t uf_find(Atomic<uint32_t> *uf, uint32_t n) {
    for (uint32_t next = uf_load(uf, n); next != n; next = uf_load(uf, n)) {
        n = next;
    }
    return n;
}

public void uf_find_compress(Atomic<uint32_t> *uf, uint32_t n) {
    const uint32_t orig = n;
    for (uint32_t next = uf_load(uf, n); next != n; next = uf_load(uf, n)) {
        n = next;
        uf[orig].min(n, MemoryOrder.Relaxed);
    }
}

public void uf_merge(Atomic<uint32_t> *uf, uint32_t a, uint32_t b) {
    while (true) {
        a = uf_find(uf, a);
        b = uf_find(uf, b);
        if (a == b) {
            break;
        }

        const uint32_t hi = max(a, b);
        const uint32_t lo = min(a, b);
        const uint32_t old = uf[hi].min(lo, MemoryOrder.Relaxed);
        if (old == hi) {
            break;
        }
        a = old;
        b = lo;
    }
}

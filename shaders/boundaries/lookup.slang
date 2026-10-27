#ifndef BOUNDARIES_LOOKUP_SLANG
#define BOUNDARIES_LOOKUP_SLANG

#include "shared/boundaries.h"

// lowerbias32, a 32-bit integer hash from the hash-prospector project.
uint32_t lowerbias32(uint32_t x) {
    x ^= x >> 16;
    x *= 0xa812d533u;
    x ^= x >> 15;
    x *= 0xb278e4adu;
    x ^= x >> 17;
    return x;
}

uint32_t hash_key(uint64_t key) {
    return lowerbias32(uint32_t(key >> 32) ^ lowerbias32(uint32_t(key)));
}

// Returns the index holding the key, inserting it if needed, or 0 if the map
// is full. Index 0 is a sentinel and not used.
uint32_t lookup(Atomic<uint32_t> *key_words, uint32_t slots, uint64_t key) {
    const uint32_t key_lo = uint32_t(key >> 32) + 1;
    const uint32_t key_hi = uint32_t(key);
    const uint32_t first_index = hash_key(key) & (slots - 1);
    uint32_t index = first_index;

    // We should be careful to have low load factors
    // or this will degrade quite terribly.
    do {
        if (index != 0) {
            const uint32_t seen_hi = key_words[2 * index].load();
            const uint32_t seen_lo = key_words[2 * index + 1].load();
            if (seen_lo == key_lo && seen_hi == key_hi) {
                return index;
            }
            if ((seen_lo != 0 && seen_lo != key_lo) ||
                (seen_hi != 0 && seen_hi != key_hi)) {
                index = (index + 1) & (slots - 1);
                continue;
            }
            const uint32_t prev_lo =
                key_words[2 * index + 1].compareExchange(0, key_lo);
            if (prev_lo == 0 || prev_lo == key_lo) {
                const uint32_t prev_hi =
                    key_words[2 * index].compareExchange(0, key_hi);
                if (prev_hi == 0 || prev_hi == key_hi) {
                    return index;
                }
            }
        }
        index = (index + 1) & (slots - 1);
    } while (index != first_index);

    return 0;
}

#endif

import shared;
import union_find;

[vk::push_constant]
ConstantBuffer<SegmentationPush> pc;

[vk::constant_id(0)]
const uint32_t RUN = 64;

groupshared uint32_t run_labels_255[RUN];
groupshared uint32_t run_counts_255[RUN];
groupshared uint32_t run_labels_0[RUN];
groupshared uint32_t run_counts_0[RUN];

uint32_t run_label(bool zeros, uint32_t i) {
    return zeros ? run_labels_0[i] : run_labels_255[i];
}

uint32_t run_count(bool zeros, uint32_t i) {
    return zeros ? run_counts_0[i] : run_counts_255[i];
}

void add_run(Atomic<uint32_t> *sizes, bool zeros, uint32_t lid, uint32_t label,
             uint32_t count) {
    if (count == 0) {
        return;
    }
    if (lid > 0 && run_count(zeros, lid - 1) > 0 &&
        run_label(zeros, lid - 1) == label) {
        return;
    }
    uint32_t total = 0;
    for (uint32_t i = lid;
         i < RUN && run_count(zeros, i) > 0 && run_label(zeros, i) == label;
         i++) {
        total += run_count(zeros, i);
    }
    sizes[label].add(total, MemoryOrder.Relaxed);
}

uint32_t pixel_label(uint32_t info, uint32_t bit_255, uint32_t bit_0,
                     uint32_t label_255, uint32_t label_0) {
    if ((info & bit_255) != 0) {
        return LABEL_PIXEL_MASK | label_255;
    }
    if ((info & bit_0) != 0) {
        return label_0;
    }
    return 0;
}

[shader("compute")]
[numthreads(RUN, 1, 1)]
void main(uint3 global_id: SV_DispatchThreadID,
          uint3 local_id: SV_GroupThreadID) {
    uint32_t *scratch = (uint32_t *)pc.scratch;
    uint32_t *labels = (uint32_t *)pc.labels;
    Atomic<uint32_t> *sizes = (Atomic<uint32_t> *)pc.sizes;

    const uint32_t width = pc.width;
    const uint32_t lid = local_id.x;
    const uint32_t block_x = global_id.x;

    const bool in_bounds = block_x < width / 2;
    const uint32_t id = global_id.y * 2 * width + block_x * 2;

    uint32_t label_255 = 0;
    uint32_t label_0 = 0;
    uint32_t info = 0;
    if (in_bounds) {
        label_255 = scratch[id];
        label_0 = scratch[id + 1];
        info = scratch[id + width];
    }
    uint32_t count_255 = countbits(info & BITMASK_POS_255);
    uint32_t count_0 = countbits(info & BITMASK_POS_0);

    // A split block's bottom 0 has a separate node in slot 3.
    // This is not visited by the compression pass, so do it here.
    uint32_t label_0_bottom = label_0;
    if ((info & SPLIT_0) != 0) {
        label_0_bottom =
            uf_find((Atomic<uint32_t> *)pc.scratch, id + width + 1);
        count_0 = 1;
        sizes[label_0_bottom].add(1, MemoryOrder.Relaxed);
    }

    run_labels_255[lid] = label_255;
    run_counts_255[lid] = count_255;
    run_labels_0[lid] = label_0;
    run_counts_0[lid] = count_0;
    GroupMemoryBarrierWithGroupSync();

    add_run(sizes, false, lid, label_255, count_255);
    add_run(sizes, true, lid, label_0, count_0);

    if (!in_bounds) {
        return;
    }
    labels[id] =
        pixel_label(info, TOP_LEFT_255, TOP_LEFT_0, label_255, label_0);
    labels[id + 1] =
        pixel_label(info, TOP_RIGHT_255, TOP_RIGHT_0, label_255, label_0);
    labels[id + width] = pixel_label(info, BOTTOM_LEFT_255, BOTTOM_LEFT_0,
                                     label_255, label_0_bottom);
    labels[id + width + 1] = pixel_label(info, BOTTOM_RIGHT_255, BOTTOM_RIGHT_0,
                                         label_255, label_0_bottom);
}

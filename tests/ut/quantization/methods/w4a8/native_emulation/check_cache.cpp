// SPDX-License-Identifier: Apache-2.0
// CPU semantics A/B for the native FP4 expert-cache LRU slot map. This is NOT
// an NPU compile/accuracy gate; it validates ComputeCacheMap against a scalar
// reference and hand-derived expected values so the deterministic, race-free
// LRU invariants are pinned before the kernel is written.
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "fp4_cache_lru.h"
#include "fp4_vector_decode.h"

namespace {
using AscendC::GlobalTensor;

struct Ref {
    std::vector<int32_t> slot;
    std::vector<int32_t> resident;
    std::vector<int32_t> finalState;
    std::vector<int32_t> finalTimestamps;
    uint64_t groupReads = 0;
    uint64_t stateReads = 0;
    uint64_t timestampReads = 0;
};

int64_t refCount(const std::vector<int64_t> &groups, uint32_t groupType, uint32_t e)
{
    int64_t v = groups[e];
    return groupType == 0 ? v - (e == 0 ? 0 : groups[e - 1]) : v;
}

// Independent scalar reference implementing the same deterministic LRU
// semantics (hit-slot pinning, first-empty-then-LRU eviction, pinned shared
// prefix, final-occupant write-through).
Ref referenceLru(const std::vector<int64_t> &groups, uint32_t experts, uint32_t groupType,
    const std::vector<int32_t> &state, const std::vector<int32_t> &ts,
    uint32_t capacity, uint32_t hitThreshold, uint32_t step, uint32_t numShared)
{
    Ref r;
    r.slot.assign(experts, -1);
    r.resident.assign(experts, 0);
    std::vector<int32_t> ws(capacity), wt(capacity);
    for (uint32_t s = 0; s < capacity; ++s) {
        ws[s] = state[s];
        wt[s] = ts[s];
    }
    for (uint32_t e = 0; e < experts; ++e) {
        if (e < numShared) {
            r.slot[e] = static_cast<int32_t>(e);
            r.resident[e] = (state[e] == static_cast<int32_t>(e)) ? 1 : 0;
            continue;
        }
        int64_t c = refCount(groups, groupType, e);
        if (c < static_cast<int64_t>(hitThreshold)) continue;
        for (uint32_t s = 0; s < capacity; ++s) {
            if (state[s] == static_cast<int32_t>(e)) {
                r.resident[e] = 1;
                r.slot[e] = static_cast<int32_t>(s);
                break;
            }
        }
    }
    for (uint32_t e = 0; e < experts; ++e) {
        if (e < numShared) {
            if (!r.resident[e]) {
                ws[e] = static_cast<int32_t>(e);
                wt[e] = static_cast<int32_t>(step);
            }
            continue;
        }
        int64_t c = refCount(groups, groupType, e);
        if (c < static_cast<int64_t>(hitThreshold)) continue;
        if (r.resident[e]) {
            wt[r.slot[e]] = static_cast<int32_t>(step);
            continue;
        }
        int32_t bestEmpty = -1, bestIdx = -1, bestTs = INT32_MAX;
        for (uint32_t s = numShared; s < capacity; ++s) {
            int32_t occ = state[s];
            if (occ >= 0 && r.resident[static_cast<uint32_t>(occ)] == 1) continue;
            if (ws[s] == -1) {
                bestEmpty = static_cast<int32_t>(s);
                break;
            }
            if (wt[s] < bestTs) {
                bestTs = wt[s];
                bestIdx = static_cast<int32_t>(s);
            }
        }
        int32_t slot = bestEmpty != -1 ? bestEmpty : bestIdx;
        if (slot != -1) {
            r.slot[e] = slot;
            ws[slot] = static_cast<int32_t>(e);
            wt[slot] = static_cast<int32_t>(step);
        }
    }
    r.finalState = ws;
    r.finalTimestamps = wt;
    return r;
}

Ref runMap(const std::vector<int64_t> &groups, uint32_t experts, uint32_t groupType,
    const std::vector<int32_t> &state, const std::vector<int32_t> &ts,
    uint32_t capacity, uint32_t hitThreshold, uint32_t step, uint32_t numShared)
{
    std::vector<int64_t> g = groups;
    std::vector<int32_t> st = state, t = ts;
    GlobalTensor<int64_t> gG;
    gG.SetGlobalBuffer(g.data());
    GlobalTensor<int32_t> sG;
    sG.SetGlobalBuffer(st.data());
    GlobalTensor<int32_t> tG;
    tG.SetGlobalBuffer(t.data());
    Fp4Int8V2::ExpertSlotMap map;
    Fp4Int8V2::ComputeCacheMap(gG, experts, groupType, sG, tG, capacity, hitThreshold, step, numShared, map);
    Ref r;
    r.slot.assign(map.slot, map.slot + experts);
    r.resident.assign(map.resident, map.resident + experts);
    r.finalState.assign(map.finalState, map.finalState + capacity);
    r.finalTimestamps.assign(map.finalTimestamps, map.finalTimestamps + capacity);
    r.groupReads = gG.scalarReads;
    r.stateReads = sG.scalarReads;
    r.timestampReads = tG.scalarReads;
    return r;
}

bool same(const Ref &a, const Ref &b, std::string &why)
{
    if (a.slot != b.slot) {
        why = "slot mismatch";
        return false;
    }
    if (a.resident != b.resident) {
        why = "resident mismatch";
        return false;
    }
    if (a.finalState != b.finalState) {
        why = "finalState mismatch";
        return false;
    }
    if (a.finalTimestamps != b.finalTimestamps) {
        why = "finalTimestamps mismatch";
        return false;
    }
    return true;
}

void dump(const char *label, const Ref &r, uint32_t experts, uint32_t capacity)
{
    std::cerr << label << " slot=[";
    for (uint32_t e = 0; e < experts; ++e) std::cerr << (e ? "," : "") << r.slot[e];
    std::cerr << "] resident=[";
    for (uint32_t e = 0; e < experts; ++e) std::cerr << (e ? "," : "") << r.resident[e];
    std::cerr << "] finalState=[";
    for (uint32_t s = 0; s < capacity; ++s) std::cerr << (s ? "," : "") << r.finalState[s];
    std::cerr << "] finalTimestamps=[";
    for (uint32_t s = 0; s < capacity; ++s) std::cerr << (s ? "," : "") << r.finalTimestamps[s];
    std::cerr << "]\n";
}

// Hand-derived expected values (the normative spec). Each case pins one LRU
// invariant: first-empty eviction, hit touch, hit-slot pinning under pressure,
// shared prefix, final-occupant write-through, cold/all-cold.
int fixedCases()
{
    struct Case {
        uint32_t experts, groupType, capacity, hitThreshold, step, numShared;
        std::vector<int64_t> groups;
        std::vector<int32_t> state, ts;
        std::vector<int32_t> slot, resident, finalState, finalTimestamps;
    };
    const std::vector<Case> cases = {
        {5, 1, 3, 2, 10, 0, {0, 2, 5, 3, 1}, {-1, -1, -1}, {0, 0, 0},
            {-1, 0, 1, 2, -1}, {0, 0, 0, 0, 0}, {1, 2, 3}, {10, 10, 10}},
        {5, 1, 3, 2, 20, 0, {0, 2, 5, 3, 1}, {1, 2, 3}, {0, 0, 0},
            {-1, 0, 1, 2, -1}, {0, 1, 1, 1, 0}, {1, 2, 3}, {20, 20, 20}},
        {5, 1, 3, 2, 30, 0, {0, 2, 5, 3, 4}, {1, 2, 3}, {0, 0, 0},
            {-1, 0, 1, 2, -1}, {0, 1, 1, 1, 0}, {1, 2, 3}, {30, 30, 30}},
        {4, 1, 4, 1, 40, 2, {10, 10, 5, 5}, {-1, -1, -1, -1}, {0, 0, 0, 0},
            {0, 1, 2, 3}, {0, 0, 0, 0}, {0, 1, 2, 3}, {40, 40, 40, 40}},
        {5, 1, 2, 2, 50, 0, {0, 0, 3, 4, 5}, {-1, -1}, {0, 0},
            {-1, -1, 0, 1, 0}, {0, 0, 0, 0, 0}, {4, 3}, {50, 50}},
        {5, 1, 3, 2, 60, 0, {0, 0, 0, 0, 0}, {1, 2, 3}, {7, 8, 9},
            {-1, -1, -1, -1, -1}, {0, 0, 0, 0, 0}, {1, 2, 3}, {7, 8, 9}},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case &c = cases[i];
        Ref got = runMap(c.groups, c.experts, c.groupType, c.state, c.ts,
            c.capacity, c.hitThreshold, c.step, c.numShared);
        Ref want;
        want.slot = c.slot;
        want.resident = c.resident;
        want.finalState = c.finalState;
        want.finalTimestamps = c.finalTimestamps;
        std::string why;
        if (!same(got, want, why)) {
            std::cerr << "fixed case " << i << " FAIL: " << why << '\n';
            dump("  got ", got, c.experts, c.capacity);
            dump("  want", want, c.experts, c.capacity);
            return 1;
        }
    }
    std::cout << "fixed LRU cases PASS: 6 hand-derived slot maps\n";
    return 0;
}

int randomizedSweep()
{
    std::mt19937 rng(20261023);
    for (uint32_t trial = 0; trial < 20000; ++trial) {
        uint32_t experts = 1 + rng() % 8;
        uint32_t capacity = 1 + rng() % 6;
        uint32_t numShared = rng() % (capacity + 1);
        uint32_t groupType = rng() % 2;
        uint32_t hitThreshold = 1 + rng() % 5;
        uint32_t step = 1 + rng() % 100;
        std::vector<int64_t> counts(experts);
        for (auto &v : counts) v = rng() % 7;
        std::vector<int64_t> groups(experts);
        int64_t acc = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            acc += counts[e];
            groups[e] = groupType == 0 ? acc : counts[e];
        }
        std::vector<int32_t> state(capacity), ts(capacity);
        for (uint32_t s = 0; s < capacity; ++s) {
            state[s] = (rng() % (experts + 1)) - 1; // -1..experts-1, allow duplicates
            ts[s] = rng() % 21;
        }
        Ref got = runMap(groups, experts, groupType, state, ts, capacity, hitThreshold, step, numShared);
        Ref want = referenceLru(groups, experts, groupType, state, ts, capacity, hitThreshold, step, numShared);
        std::string why;
        if (!same(got, want, why)) {
            std::cerr << "randomized trial " << trial << " FAIL: " << why
                << " experts=" << experts << " capacity=" << capacity
                << " numShared=" << numShared << " groupType=" << groupType
                << " threshold=" << hitThreshold << " step=" << step << '\n';
            dump("  got ", got, experts, capacity);
            dump("  want", want, experts, capacity);
            return 1;
        }
    }
    std::cout << "randomized LRU sweep PASS: 20000 configs vs scalar reference\n";
    return 0;
}

// Exercise real E=256 metadata sizes and assert the reduction in scalar GM
// reads independently of CPU/NPU timing. The unchanged reference is the oracle
// for every output field, including duplicate occupants and shared slots.
int largeMetadataSweep()
{
    std::mt19937 rng(20261027);
    constexpr uint32_t experts = Fp4Int8V2::MAX_CACHE_SLOTS;
    for (uint32_t trial = 0; trial < 2000; ++trial) {
        uint32_t capacity = 1 + rng() % experts;
        uint32_t numShared = rng() % (std::min(capacity, 4U) + 1);
        uint32_t groupType = rng() % 2;
        uint32_t threshold = 1 + rng() % 4;
        std::vector<int64_t> groups(experts);
        int64_t acc = 0;
        for (uint32_t e = 0; e < experts; ++e) {
            int64_t count = rng() % 7;
            acc += count;
            groups[e] = groupType == 0 ? acc : count;
        }
        std::vector<int32_t> state(capacity), ts(capacity);
        for (uint32_t s = 0; s < capacity; ++s) {
            state[s] = static_cast<int32_t>(rng() % (experts + 1)) - 1;
            ts[s] = rng() % 101;
        }
        Ref got = runMap(groups, experts, groupType, state, ts, capacity, threshold, 102, numShared);
        Ref want = referenceLru(groups, experts, groupType, state, ts, capacity, threshold, 102, numShared);
        std::string why;
        if (!same(got, want, why)) {
            std::cerr << "large LRU trial " << trial << " FAIL: " << why << '\n';
            return 1;
        }
        uint64_t expectedGroupReads = experts - numShared + (groupType == 0 && numShared > 0 ? 1 : 0);
        if (got.stateReads != capacity || got.timestampReads != capacity || got.groupReads != expectedGroupReads) {
            std::cerr << "metadata GM reads are not linear: state=" << got.stateReads
                << " timestamps=" << got.timestampReads << " groups=" << got.groupReads << '\n';
            return 1;
        }
    }
    std::cout << "large LRU metadata sweep PASS: 2000 E=256 configs, exact maps and linear GM reads\n";
    return 0;
}

// Cache-fill bit-exactness + hit-copy identity. The INT8 written into a persistent
// [N,K] cache slot by StoreDecodedRows must be byte-identical to an on-the-fly decode
// of the same rows, and LoadDecodedRows must round-trip those exact bytes back into UB
// (the hit path's only data movement). Exercises the new GM->UB strided load.
int cacheFillBitExact()
{
    using namespace AscendC;
    constexpr uint32_t NROWS = 32;
    constexpr uint32_t PREFIX = 32;
    for (uint32_t k : {256U, 512U, 4096U, 8192U}) {
        const uint32_t tileK = std::min(k, Fp4Int8V2::TILE_K);
        const uint32_t blocks = tileK / Fp4Int8V2::SCALE_BLOCK;
        TPipe pipe;
        Fp4Int8V2::VectorDecoder decoder;
        decoder.Init(pipe);
        TBuf<QuePosition::VECCALC> packedBuf, outputBuf, channelBuf;
        pipe.InitBuffer(packedBuf, Fp4Int8V2::DECODE_ROWS * tileK / 2);
        pipe.InitBuffer(outputBuf, Fp4Int8V2::DECODE_ROWS * tileK);
        pipe.InitBuffer(channelBuf, Fp4Int8V2::TILE_SCALE_ROWS * sizeof(float));
        auto packed = packedBuf.Get<uint8_t>();
        auto output = outputBuf.Get<int8_t>();
        auto channels = channelBuf.Get<float>();
        for (uint32_t row = 0; row < Fp4Int8V2::TILE_SCALE_ROWS; ++row)
            channels.Write(row, 0.5f + row * 0.03125f);

        std::vector<uint8_t> bytes(PREFIX + NROWS * k / 2 + 32, 0xEE);
        std::vector<uint8_t> raw(PREFIX + NROWS * k / 32 + 32, 0xEE);
        for (uint32_t row = 0; row < NROWS; ++row) {
            for (uint32_t i = 0; i < k / 2; ++i)
                bytes[PREFIX + row * k / 2 + i] = (row * 19 + i) % 256;
            for (uint32_t b = 0; b < k / 32; ++b)
                raw[PREFIX + row * k / 32 + b] = b % 5 == 0 ? 0 : 118 + (row + b) % 15;
        }
        GlobalTensor<uint8_t> weights, scales;
        weights.SetGlobalBuffer(bytes.data());
        scales.SetGlobalBuffer(raw.data());

        // One persistent [N,K] INT8 cache slot + one independent on-the-fly buffer.
        std::vector<int8_t> cache(NROWS * k, -1);
        std::vector<int8_t> onthefly(NROWS * k, -1);
        GlobalTensor<int8_t> cacheG, ontheflyG;
        cacheG.SetGlobalBuffer(cache.data());
        ontheflyG.SetGlobalBuffer(onthefly.data());

        for (uint32_t offset = 0; offset < k; offset += tileK) {
            decoder.PrecomputeTileScale(scales, PREFIX + offset / 32, k / 32, blocks, channels);
            for (uint32_t row = 0; row < Fp4Int8V2::TILE_SCALE_ROWS; row += Fp4Int8V2::DECODE_ROWS) {
                decoder.LoadPackedRows(packed, weights[PREFIX + row * k / 2 + offset / 2], k / 2, tileK);
                // miss path: decode + write-through into the cache slot.
                decoder.DecodeRowsPrecomputed(packed, row, output, tileK);
                decoder.StoreDecodedRows(cacheG[row * k + offset], output, k, tileK);
                // independent on-the-fly decode into a separate buffer.
                decoder.DecodeRowsPrecomputed(packed, row, output, tileK);
                decoder.StoreDecodedRows(ontheflyG[row * k + offset], output, k, tileK);
            }
        }
        // (1) cache fill == on-the-fly decode, byte-for-byte.
        for (uint32_t i = 0; i < NROWS * k; ++i) {
            if (cache[i] != onthefly[i]) {
                std::cerr << "cache-fill mismatch K=" << k << " i=" << i
                    << " cache=" << static_cast<int>(cache[i])
                    << " onthefly=" << static_cast<int>(onthefly[i]) << '\n';
                return 1;
            }
        }
        // (2) hit-copy identity: LoadDecodedRows round-trips the exact cache bytes.
        for (uint32_t offset = 0; offset < k; offset += tileK) {
            for (uint32_t row = 0; row < Fp4Int8V2::TILE_SCALE_ROWS; row += Fp4Int8V2::DECODE_ROWS) {
                decoder.LoadDecodedRows(output, cacheG[row * k + offset], k, tileK);
                for (uint32_t e = 0; e < Fp4Int8V2::DECODE_ROWS * tileK; ++e) {
                    int8_t expected = cache[row * k + offset + (e / tileK) * k + (e % tileK)];
                    if (output.Read(e) != expected) {
                        std::cerr << "hit-copy mismatch K=" << k << " offset=" << offset
                            << " row=" << row << " e=" << e
                            << " loaded=" << static_cast<int>(output.Read(e))
                            << " cache=" << static_cast<int>(expected) << '\n';
                        return 1;
                    }
                }
            }
        }
    }
    std::cout << "cache-fill bit-exact + hit-copy identity PASS: K=256/512/4096/8192\n";
    return 0;
}
} // namespace

int main()
{
    if (fixedCases()) return 1;
    if (randomizedSweep()) return 1;
    if (largeMetadataSweep()) return 1;
    if (cacheFillBitExact()) return 1;
    std::cout << "native FP4 expert-cache LRU slot map PASS\n";
    return 0;
}

// SPDX-License-Identifier: Apache-2.0
// CPU semantics shim for the real decoder header. This is NOT an Ascend
// compiler, pipeline simulator, or proof of NPU instruction support.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#define __aicore__
#define __gm__
using half = _Float16;
constexpr int PIPE_V = 0;
constexpr int EVENT_ID7 = 7;
namespace AscendC {
// BF16 实验方法与 INT8 共用正式头文件；保持独立的 16-bit 存储，禁止冒充 half。
struct bfloat16_t {
    uint16_t bits;
    bfloat16_t() = default;
    explicit bfloat16_t(float value) {
        uint32_t word;
        std::memcpy(&word, &value, sizeof(word));
        if ((word & 0x7fffffffU) > 0x7f800000U) {
            bits = static_cast<uint16_t>((word >> 16) | 0x0040U);
        } else {
            word += 0x7fffU + ((word >> 16) & 1U);
            bits = static_cast<uint16_t>(word >> 16);
        }
    }
    operator float() const {
        uint32_t word = static_cast<uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &word, sizeof(value));
        return value;
    }
};
static_assert(sizeof(bfloat16_t) == 2);
struct int4b_t {};
enum class QuePosition { VECCALC, VECIN, VECOUT };
enum class RoundMode { CAST_NONE, CAST_TRUNC, CAST_RINT, CAST_ODD };
enum class CMPMODE { NE, EQ, GE, LT };
enum class SELMODE { VSEL_TENSOR_SCALAR_MODE };
enum class HardEvent { MTE2_V, V_MTE2, S_V };
// 只统计源码级 API 调用，便于结构对照；不模拟 NPU 指令、周期，也不据此判断计算或搬运瓶颈。
inline uint64_t g_vecCount = 0;
inline uint64_t g_barrierCount = 0;
inline uint64_t g_dmaCount = 0;
inline void ResetCounters() { g_vecCount = g_barrierCount = g_dmaCount = 0; }
template <class T> class LocalTensor {
public:
    std::shared_ptr<std::vector<uint8_t>> storage;
    size_t offset = 0;
    LocalTensor operator[](size_t index) const {
        auto result = *this;
        result.offset += index * sizeof(T);
        return result;
    }
    template <class U> LocalTensor<U> ReinterpretCast() const { return {storage, offset}; }
    double Read(size_t index) const {
        if constexpr (std::is_same_v<T, int4b_t>) {
            size_t at = offset + index / 2;
            if (at >= storage->size()) throw std::out_of_range("int4 read");
            int value = ((*storage)[at] >> (4 * (index % 2))) & 15;
            return value >= 8 ? value - 16 : value;
        } else {
            size_t at = offset + index * sizeof(T);
            if (at + sizeof(T) > storage->size()) throw std::out_of_range("tensor read");
            T value;
            std::memcpy(&value, storage->data() + at, sizeof(T));
            return static_cast<double>(value);
        }
    }
    void Write(size_t index, double value) const {
        size_t at = offset + index * sizeof(T);
        if (at + sizeof(T) > storage->size()) throw std::out_of_range("tensor write");
        T typed = static_cast<T>(value);
        std::memcpy(storage->data() + at, &typed, sizeof(T));
    }
    void SetValue(size_t index, T value) const { Write(index, value); }
    T GetValue(size_t index) const { return static_cast<T>(Read(index)); }
};
template <class T> struct GlobalTensor {
    T *data;
    mutable uint64_t scalarReads = 0;
    void SetGlobalBuffer(T *p) { data = p; }
    GlobalTensor operator[](size_t index) const { return {data + index}; }
    // Scalar GM access, mirroring the AscendC GlobalTensor scalar API used by
    // the cache LRU bookkeeping (GetValue/SetValue). Needed so fp4_cache_lru.h
    // compiles unchanged against both the CANN headers and this CPU shim.
    T GetValue(size_t index) const { ++scalarReads; return data[index]; }
    void SetValue(size_t index, T value) { data[index] = value; }
};
template <QuePosition> struct TBuf {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    template <class T> LocalTensor<T> Get() { return {bytes, 0}; }
};
// TQue 最小语义：只模拟槽位分配与 FIFO 顺序，不模拟 MTE2/Vector/MTE3 时序。
// AllocTensor 槽满即抛错（捕获预取超用 STAGES 的 bug）；DeQue 空队列抛错（捕获 EnQue/DeQue 失衡）。
// 残余在途（EnQue 后未被 DeQue 消费）通过 fifo 非空暴露，用于检测「最后一批/最后 tile 仍越界预取」。
template <QuePosition P, uint32_t STAGES>
struct TQue {
    struct Slot { std::shared_ptr<std::vector<uint8_t>> bytes; bool inUse = false; };
    std::vector<Slot> slots;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> fifo;
    void Init(uint32_t numBuffers, size_t bufferBytes) {
        slots.assign(numBuffers, Slot{});
        for (auto &s : slots) s.bytes = std::make_shared<std::vector<uint8_t>>(bufferBytes);
    }
    template <class T> LocalTensor<T> AllocTensor() {
        for (auto &s : slots)
            if (!s.inUse) { s.inUse = true; return LocalTensor<T>{s.bytes, 0}; }
        throw std::runtime_error("TQue AllocTensor: all slots busy (over-committed)");
    }
    template <class T> void EnQue(LocalTensor<T> t) {
        for (auto &s : slots)
            if (s.bytes == t.storage) { fifo.push_back(s.bytes); return; }
        throw std::runtime_error("TQue EnQue: unknown tensor");
    }
    template <class T> LocalTensor<T> DeQue() {
        if (fifo.empty()) throw std::runtime_error("TQue DeQue: empty queue");
        auto bytes = fifo.front();
        fifo.erase(fifo.begin());
        return LocalTensor<T>{bytes, 0};
    }
    template <class T> void FreeTensor(LocalTensor<T> t) {
        for (auto &s : slots)
            if (s.bytes == t.storage) { s.inUse = false; return; }
        throw std::runtime_error("TQue FreeTensor: unknown tensor");
    }
};
struct TPipe {
    template <QuePosition P> void InitBuffer(TBuf<P> &buffer, size_t bytes) {
        buffer.bytes = std::make_shared<std::vector<uint8_t>>(bytes);
    }
    template <QuePosition P, uint32_t S> void InitBuffer(TQue<P, S> &queue, uint32_t numBuffers, size_t bufferBytes) {
        queue.Init(numBuffers, bufferBytes);
    }
};
template <int> void PipeBarrier() { ++g_barrierCount; }
template <HardEvent> void SetFlag(int) {}
template <HardEvent> void WaitFlag(int) {}

// 仅用于候选的有界有限 FP32 输入：先向零截到 FP16，若舍弃位非零则将最低位设为奇数。
// 范围外直接拒绝，避免把不完整的 CPU 浮点模拟当成 CANN 全类型/全数域的实现。
inline half HalfRoundOdd(float value) {
    if (!std::isfinite(value) || std::fabs(value) > 127.f)
        throw std::domain_error("round-odd shim supports only finite [-127,127]");
    half rounded = static_cast<half>(value);
    if (static_cast<float>(rounded) == value) return rounded;
    uint16_t bits;
    std::memcpy(&bits, &rounded, sizeof(bits));
    if (std::fabs(static_cast<float>(rounded)) > std::fabs(value)) --bits;
    bits |= 1;
    std::memcpy(&rounded, &bits, sizeof(bits));
    return rounded;
}
template <class D, class S> void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode mode, uint32_t count) {
    // Only the type pairs used by this decoder and supported by Atlas A2.
    // A generic C++ numeric cast hid the unsupported int16 -> int32 path.
    // This subset is not a replacement for compiling against CANN headers.
    constexpr bool supported =
        (std::is_same_v<D, bfloat16_t> && std::is_same_v<S, float>) ||
        (std::is_same_v<D, half> &&
            (std::is_same_v<S, int4b_t> || std::is_same_v<S, uint8_t> || std::is_same_v<S, float>)) ||
        (std::is_same_v<D, float> &&
            (std::is_same_v<S, int16_t> || std::is_same_v<S, int32_t> || std::is_same_v<S, half> || std::is_same_v<S, float>)) ||
        (std::is_same_v<S, half> &&
            (std::is_same_v<D, int16_t> || std::is_same_v<D, int32_t> || std::is_same_v<D, int8_t>));
    static_assert(supported, "unsupported Atlas A2 decoder Cast type pair");
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) {
        double value = src.Read(i);
        if constexpr (std::is_same_v<D, bfloat16_t> && std::is_same_v<S, float>) {
            if (mode != RoundMode::CAST_RINT) {
                throw std::invalid_argument("BF16 shim only implements float to bf16 RINT");
            }
            dst.Write(i, value);
            continue;
        }
        if constexpr (std::is_same_v<D, half> && std::is_same_v<S, float>) {
            // FP32→FP16 的 RINT 舍入目标是 FP16 格点，不能提前 nearbyint 到整数。
            if (mode == RoundMode::CAST_RINT || mode == RoundMode::CAST_NONE) {
                dst.Write(i, value);
                continue;
            }
        }
        if (mode == RoundMode::CAST_ODD) {
            if constexpr (std::is_same_v<D, half> && std::is_same_v<S, float>) {
                dst.Write(i, static_cast<float>(HalfRoundOdd(static_cast<float>(value))));
                continue;
            } else {
                throw std::invalid_argument("round-odd shim only supports float to half");
            }
        }
        if (mode == RoundMode::CAST_RINT) value = std::nearbyint(value);
        if constexpr (std::is_same_v<S, half> && std::is_same_v<D, int8_t>) {
            if (mode == RoundMode::CAST_NONE) value = std::nearbyint(value);
        }
        if (mode == RoundMode::CAST_TRUNC) value = std::trunc(value);
        dst.Write(i, value);
    }
}
template <class T, class S> void Duplicate(LocalTensor<T> dst, S value, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, value);
}
template <class T> void And(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, static_cast<T>(a.Read(i)) & static_cast<T>(b.Read(i)));
}
// Match the same-type scalar deduction in the CANN overload. An independently
// typed uint32_t argument here used to hide the signed-literal compile error.
template <class T> void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const T& bits, const int32_t& count) {
    ++g_vecCount;
    for (int32_t i = 0; i < count; ++i) dst.Write(i, static_cast<T>(src.Read(i)) << bits);
}
template <class T> void CompareScalar(LocalTensor<uint8_t> mask, LocalTensor<T> src, T value,
    CMPMODE mode, uint32_t count) {
    std::vector<bool> flags(count);
    for (uint32_t i = 0; i < count; ++i) {
        double x = src.Read(i);
        flags[i] = mode == CMPMODE::NE ? x != value : mode == CMPMODE::EQ ? x == value :
            mode == CMPMODE::GE ? x >= value : x < value;
    }
    for (uint32_t i = 0; i < count; i += 8) {
        uint8_t bits = 0;
        for (uint32_t j = 0; j < 8 && i + j < count; ++j) bits |= flags[i + j] << j;
        mask.Write(i / 8, bits);
    }
}
template <class T> void Select(LocalTensor<T> dst, LocalTensor<uint8_t> mask, LocalTensor<T> yes,
    T no, SELMODE, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        bool predicate = (static_cast<uint8_t>(mask.Read(i / 8)) >> (i % 8)) & 1;
        dst.Write(i, predicate ? yes.Read(i) : no);
    }
}
template <class T> void Add(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, static_cast<float>(a.Read(i)) + static_cast<float>(b.Read(i)));
}
template <class T> void Adds(LocalTensor<T> dst, LocalTensor<T> src, T value, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, src.Read(i) + value);
}
template <class T> void Muls(LocalTensor<T> dst, LocalTensor<T> src, T value, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, src.Read(i) * value);
}
template <class T> void Gather(LocalTensor<T> dst, LocalTensor<T> table,
    LocalTensor<uint32_t> offsets, uint32_t base, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, table.Read((base + offsets.Read(i)) / sizeof(T)));
}
template <class T> void Mul(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, static_cast<float>(a.Read(i)) * static_cast<float>(b.Read(i)));
}
struct BinaryRepeatParams {
    uint8_t dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride;
};
template <class T> void Mul(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b,
    uint64_t mask, uint8_t repeats, BinaryRepeatParams stride) {
    ++g_vecCount;
    constexpr uint32_t lanes = 32 / sizeof(T);
    if (mask > 256 / sizeof(T)) throw std::invalid_argument("vector mask exceeds one repeat");
    for (uint32_t repeat = 0; repeat < repeats; ++repeat)
        for (uint32_t lane = 0; lane < mask; ++lane) {
            size_t at = (repeat * stride.dstRepStride + lane / lanes * stride.dstBlkStride) * lanes + lane % lanes;
            size_t aAt = (repeat * stride.src0RepStride + lane / lanes * stride.src0BlkStride) * lanes + lane % lanes;
            size_t bAt = (repeat * stride.src1RepStride + lane / lanes * stride.src1BlkStride) * lanes + lane % lanes;
            dst.Write(at, static_cast<float>(a.Read(aAt)) * static_cast<float>(b.Read(bAt)));
        }
}
template <class T> void Div(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, static_cast<float>(a.Read(i)) / static_cast<float>(b.Read(i)));
}
template <class T> void Maxs(LocalTensor<T> dst, LocalTensor<T> src, T value, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, std::max(static_cast<T>(src.Read(i)), value));
}
template <class T> void Mins(LocalTensor<T> dst, LocalTensor<T> src, T value, uint32_t count) {
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, std::min(static_cast<T>(src.Read(i)), value));
}
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, rsv; };
struct DataCopyParams { uint16_t blockCount, blockLen, srcStride, dstStride; };
template <class T> void DataCopy(LocalTensor<T> dst, LocalTensor<T> src, DataCopyParams copy) {
    // A2 UB-to-UB copying is a vector-pipe operation. This shim checks
    // addresses and values only; barriers are not simulated.
    ++g_vecCount;
    const size_t length = copy.blockLen * 32 / sizeof(T);
    for (uint32_t row = 0; row < copy.blockCount; ++row)
        for (size_t i = 0; i < length; ++i)
            dst.Write(row * (length + copy.dstStride * 32 / sizeof(T)) + i,
                src.Read(row * (length + copy.srcStride * 32 / sizeof(T)) + i));
}
template <class T> void DataCopy(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyParams copy) {
    ++g_dmaCount;
    const size_t length = copy.blockLen * 32 / sizeof(T);
    for (uint32_t row = 0; row < copy.blockCount; ++row)
        for (size_t i = 0; i < length; ++i)
            dst.Write(row * (length + copy.dstStride * 32 / sizeof(T)) + i,
                src.data[row * (length + copy.srcStride * 32 / sizeof(T)) + i]);
}
template <class T> void DataCopy(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyParams copy) {
    ++g_dmaCount;
    const size_t length = copy.blockLen * 32 / sizeof(T);
    for (uint32_t row = 0; row < copy.blockCount; ++row)
        for (size_t i = 0; i < length; ++i)
            dst.data[row * (length + copy.dstStride * 32 / sizeof(T)) + i] =
                static_cast<T>(src.Read(row * (length + copy.srcStride * 32 / sizeof(T)) + i));
}
template <class T> struct DataCopyPadExtParams { bool pad; uint8_t left, right; T value; };
template <class T> void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src,
    DataCopyExtParams copy, DataCopyPadExtParams<T> pad) {
    ++g_dmaCount;
    const size_t dstPitch = ((copy.blockLen + 31) / 32 + copy.dstStride) * 32 / sizeof(T);
    const size_t srcPitch = (copy.blockLen + copy.srcStride) / sizeof(T);
    for (uint32_t row = 0; row < copy.blockCount; ++row) {
        for (uint32_t i = 0; i < copy.blockLen / sizeof(T); ++i)
            dst.Write(row * dstPitch + pad.left + i, src.data[row * srcPitch + i]);
        for (uint32_t i = 0; i < pad.right; ++i)
            dst.Write(row * dstPitch + pad.left + copy.blockLen / sizeof(T) + i, pad.value);
    }
}
// 联合查表使用的 arch22 API 子集；仅检查类型、地址和数值，不模拟硬件周期。
template <class T> void CreateVecIndex(LocalTensor<T> dst, const T &first, uint32_t count) {
    static_assert(std::is_same_v<T, int32_t>);
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i) dst.Write(i, first + static_cast<T>(i));
}
template <class T> void ShiftRight(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const T &shift, const int32_t &count) {
    static_assert(std::is_same_v<T, uint32_t>);
    ++g_vecCount;
    for (int32_t i = 0; i < count; ++i) dst.Write(i, static_cast<T>(src.Read(i)) >> shift);
}
template <class T> void Sub(LocalTensor<T> dst, LocalTensor<T> a, LocalTensor<T> b, uint32_t count) {
    static_assert(std::is_same_v<T, int32_t>);
    ++g_vecCount;
    for (uint32_t i = 0; i < count; ++i)
        dst.Write(i, static_cast<T>(a.Read(i)) - static_cast<T>(b.Read(i)));
}
struct BrcbRepeatParams { uint16_t dstBlkStride; uint16_t dstRepStride; };
template <class T> void Brcb(LocalTensor<T> dst, LocalTensor<T> src, uint8_t repeats, BrcbRepeatParams stride) {
    ++g_vecCount;
    constexpr uint32_t lanes = 32 / sizeof(T);
    for (uint32_t r = 0; r < repeats; ++r)
        for (uint32_t s = 0; s < 8; ++s)
            for (uint32_t j = 0; j < lanes; ++j)
                dst.Write((r * stride.dstRepStride + s * stride.dstBlkStride) * lanes + j, src.Read(r * 8 + s));
}
} // namespace AscendC

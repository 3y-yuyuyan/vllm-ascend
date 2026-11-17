// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_FP4_VECTOR_DECODE_H
#define VLLM_ASCEND_FP4_VECTOR_DECODE_H

#include "kernel_operator.h"

namespace Fp4Int8V2 {
using namespace AscendC;

constexpr uint32_t TILE_M = 32;
constexpr uint32_t TILE_N = 64;
constexpr uint32_t TILE_K = 512;
constexpr uint32_t SCALE_BLOCK = 32;
constexpr uint32_t STAGES = 2;
constexpr uint32_t E8M0_EXPONENT_SHIFT = 23;
constexpr uint32_t FLOATS_PER_BLOCK = 8;
constexpr uint32_t MAG_TABLE_SIZE = 16;
constexpr uint32_t DECODE_ROWS = 16;
constexpr uint32_t TILE_SCALE_ROWS = TILE_N / 2;
constexpr uint32_t DECODE_ELEMENTS = DECODE_ROWS * TILE_K;
constexpr uint32_t DMA_BLOCK_BYTES = 32;
constexpr uint32_t FP4_CODE_BITS = 4;
constexpr uint32_t PAIR_LUT_ELEMENTS = MAG_TABLE_SIZE * MAG_TABLE_SIZE;
constexpr uint32_t HALF_WORD_BITS = 16;
constexpr uint32_t UB_OPERAND_PADDING_BYTES = 256;
constexpr uint32_t SCALE_ROW_PITCH = DMA_BLOCK_BYTES;
static_assert((TILE_N / 2) % DECODE_ROWS == 0);
static_assert(TILE_K / SCALE_BLOCK <= SCALE_ROW_PITCH);
static_assert(TILE_SCALE_ROWS * SCALE_ROW_PITCH <= DECODE_ELEMENTS);

// FP4(E2M1)→BF16 解码器：E2M1 ⊂ BF16 无损，跳过 INT8 的 Maxs/Mins 限幅与多级转换，
// 仅 Gather 查表 + 尺度相乘 + 一次 float→bf16 CAST_RINT。真值权重 = fp4_value × 2^(E8M0-127)。
class VectorDecoder {
public:
    __aicore__ inline void Init(TPipe &pipe, uint32_t taskK)
    {
        // 表大小与 K 无关（256 项 × 4 B），K2048 的 taskScales 比 K4096 小 8 KiB，
        // 加 1 KiB 配对表仍低于 UB 上限；K8192 已接近上限，不放开。
        usePairLut_ = taskK == 4096 || taskK == 2048;
        if (usePairLut_) {
            // 一个字节包含两个 FP4；256 项，每项两个精确 FP16 位模式，共 1 KiB。
            // 保留 256B 尾部错位；表不含尺度，尺度始终在 FP32 中处理。
            pipe.InitBuffer(pairLut_, PAIR_LUT_ELEMENTS * sizeof(uint32_t) + UB_OPERAND_PADDING_BYTES);
        }
        pipe.InitBuffer(magTable_, MAG_TABLE_SIZE * sizeof(float));
        pipe.InitBuffer(half_, DECODE_ELEMENTS * sizeof(half));
        // 保留已验收的 index_/floats_ 相对错位；配对表缩小后需实卡复查 bank 指标。
        // Gather 用完 index_ 后，它复用为尺度广播区；两者之间有 PIPE_V 屏障。
        pipe.InitBuffer(index_, DECODE_ELEMENTS * sizeof(int32_t)
            + (usePairLut_ ? UB_OPERAND_PADDING_BYTES : 0));
        pipe.InitBuffer(floats_, DECODE_ELEMENTS * sizeof(float));
        pipe.InitBuffer(tileScale_, TILE_SCALE_ROWS * SCALE_ROW_PITCH * sizeof(float));
        // K8192 时为 32 KiB；容量随 taskK 分配，不能固定成 K4096 的 16 KiB。
        pipe.InitBuffer(taskScales_, TILE_SCALE_ROWS * (taskK / SCALE_BLOCK) * sizeof(float));

        auto table = magTable_.Get<float>();
        const float vals[MAG_TABLE_SIZE] = {
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f
        };
        for (uint32_t i = 0; i < MAG_TABLE_SIZE; ++i) {
            table.SetValue(i, vals[i]);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        if (usePairLut_) {
            BuildPairLut();
        }
    }

    // 单次跨步 DMA 搬运 16 个输出通道，每通道 tileK/2 字节；K512 时总计 4 KiB。
    __aicore__ inline void LoadPackedRows(LocalTensor<uint8_t> packed,
        GlobalTensor<uint8_t> source, uint32_t rowStride, uint32_t tileK)
    {
        DataCopyParams copy{DECODE_ROWS, static_cast<uint16_t>(tileK / 2 / DMA_BLOCK_BYTES),
            static_cast<uint16_t>((rowStride - tileK / 2) / DMA_BLOCK_BYTES), 0};
        DataCopy(packed, source, copy);
    }

    // BF16 输出每元素 2 字节，行跨步（字节）为 INT8 的 2 倍；MTE3 写回带宽翻倍。
    __aicore__ inline void StoreDecodedRowsBf16(GlobalTensor<bfloat16_t> destination,
        LocalTensor<bfloat16_t> decoded, uint32_t rowStride, uint32_t tileK)
    {
        DataCopyParams copy{DECODE_ROWS, static_cast<uint16_t>(tileK * 2 / DMA_BLOCK_BYTES), 0,
            static_cast<uint16_t>((rowStride - tileK * 2) / DMA_BLOCK_BYTES)};
        DataCopy(destination, decoded, copy);
    }

    // StoreDecodedRowsBf16 的逆操作：缓存命中路径把 bf16_cache 槽位中已解码的 BF16 行
    // 直接跨步搬运回 UB（GM→UB），与未命中路径 StoreDecodedRowsBf16 逐字节对称。
    // 行跨步（rowStride）与 StoreDecodedRowsBf16 一致，以字节为单位（t.k * 2）。
    __aicore__ inline void LoadDecodedRowsBf16(LocalTensor<bfloat16_t> decoded,
        GlobalTensor<bfloat16_t> source, uint32_t rowStride, uint32_t tileK)
    {
        DataCopyParams copy{DECODE_ROWS, static_cast<uint16_t>(tileK * 2 / DMA_BLOCK_BYTES),
            static_cast<uint16_t>((rowStride - tileK * 2) / DMA_BLOCK_BYTES), 0};
        DataCopy(decoded, source, copy);
    }

    // 每个权重任务一次搬入本 AIV 的 32 个通道尺度，GM 布局为 [32, K/32]。
    // 复用解码临时缓冲完成转换；taskScales_ 保留自然布局，tileScale_ 保存当前 tile 的紧凑视图。
    __aicore__ inline void PrecomputeTaskScalesBf16(
        const GlobalTensor<uint8_t> &scales, uint64_t scaleOffset,
        uint32_t k)
    {
        const uint32_t totalBlocks = k / SCALE_BLOCK;
        const uint32_t total = TILE_SCALE_ROWS * totalBlocks;
        auto raw = half_.Get<uint8_t>();
        auto halfCodes = index_.Get<half>();
        auto taskScales = taskScales_.Get<float>();

        // 32 个通道行每行 totalBlocks 字节，在 GM 中首尾相接，总长 k 字节、32B 对齐。
        DataCopyParams load{1, static_cast<uint16_t>(k / DMA_BLOCK_BYTES), 0, 0};
        SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        DataCopy(raw, scales[scaleOffset], load);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
        Cast(halfCodes, raw, RoundMode::CAST_NONE, total);
        PipeBarrier<PIPE_V>();
        Cast(taskScales.ReinterpretCast<int32_t>(), halfCodes, RoundMode::CAST_TRUNC, total);
        PipeBarrier<PIPE_V>();
        ShiftLeft(taskScales.ReinterpretCast<uint32_t>(), taskScales.ReinterpretCast<uint32_t>(),
            // 配对表只解包幅值；所有 K 都保留原始 FP32 尺度位模式。
            E8M0_EXPONENT_SHIFT, total);
        PipeBarrier<PIPE_V>();
    }

    // 逐 tile 将自然布局的切片压紧到 tileScale_ [32, blocks]，保持 Brcb 的连续取数约定。
    // 上个 tile 的 DecodeRowsBf16 已用 PIPE_V 结束，覆盖尺度不会踩到尚未完成的广播。
    __aicore__ inline void SelectTileScales(uint32_t tile, uint32_t k, uint32_t tileK)
    {
        const uint32_t blocks = tileK / SCALE_BLOCK;
        const uint32_t tiles = k / tileK;
        auto taskScales = taskScales_.Get<uint32_t>();
        auto tileScale = tileScale_.Get<uint32_t>();
        DataCopyParams compact{static_cast<uint16_t>(TILE_SCALE_ROWS),
            static_cast<uint16_t>(blocks / FLOATS_PER_BLOCK),
            static_cast<uint16_t>((tiles - 1) * (blocks / FLOATS_PER_BLOCK)), 0};
        DataCopy(tileScale, taskScales[static_cast<uint64_t>(tile) * blocks], compact);
        PipeBarrier<PIPE_V>();
    }

    // 原始 nibble 0..7 为正值、8..15 为负值；低 nibble 对应前一个输出。
    // 这些 E2M1 幅值都可由 FP16 精确表示，包括正负零；不能把尺度转成 FP16。
    __aicore__ inline void BuildPairLut()
    {
        const uint32_t magnitudeBits[MAG_TABLE_SIZE] = {
            0x0000U, 0x3800U, 0x3c00U, 0x3e00U, 0x4000U, 0x4200U, 0x4400U, 0x4600U,
            0x8000U, 0xb800U, 0xbc00U, 0xbe00U, 0xc000U, 0xc200U, 0xc400U, 0xc600U
        };
        auto table = pairLut_.Get<uint32_t>();
        for (uint32_t high = 0; high < MAG_TABLE_SIZE; ++high) {
            for (uint32_t low = 0; low < MAG_TABLE_SIZE; ++low) {
                table.SetValue((high << FP4_CODE_BITS) | low,
                    magnitudeBits[low] | (magnitudeBits[high] << HALF_WORD_BITS));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
    }

    __aicore__ inline void DecodeRowsPairLut(const LocalTensor<uint8_t> &packed,
        uint32_t rowBase, LocalTensor<bfloat16_t> output, uint32_t tileK, uint32_t rows)
    {
        const uint32_t count = rows * tileK;
        const uint32_t pairs = count / 2;
        const uint32_t blocks = tileK / SCALE_BLOCK;
        auto halfValues = half_.Get<half>();
        auto index = index_.Get<int32_t>();
        auto value = floats_.Get<float>();
        auto scales = index_.Get<float>();

        // 每个打包字节只生成一个索引，0..255 乘 4 得到 0..1020 字节偏移。
        // 该范围在 FP16 内精确；显式浮点字面量避免 aicore unsigned→half 转换。
        constexpr half PAIR_ENTRY_BYTES = static_cast<half>(4.0f);
        // 索引链与 Gather 全部落在同一条 PIPE_V 上，指令按序执行，RAW 依赖不需要额外屏障；
        // 这里省掉 3 次 PipeBarrier<PIPE_V>，避免把向量流水线反复排空。
        Cast(halfValues, packed, RoundMode::CAST_NONE, pairs);
        Muls(halfValues, halfValues, PAIR_ENTRY_BYTES, pairs);
        Cast(index, halfValues, RoundMode::CAST_TRUNC, pairs);

        // 一次 32-bit Gather 返回相邻的两个 FP16 幅值；只搬位模式，不做半精度尺度乘法。
        // half_ 的旧索引输入已用完，Gather 目标不与仍存活的 index_ 重叠。
        Gather(halfValues.ReinterpretCast<uint32_t>(), pairLut_.Get<uint32_t>(),
            index.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), pairs);
        Cast(value, halfValues, RoundMode::CAST_NONE, count);
        // 下一段 Brcb 会把尺度写进 index_（Gather 索引的死区），跨指令类型前保留屏障。
        PipeBarrier<PIPE_V>();

        // 索引生命周期结束后复用其空间；尺度、乘积和最终 RINT 与原 FP32 路径一致。
        for (uint32_t quarter = 0; quarter < SCALE_BLOCK / FLOATS_PER_BLOCK; ++quarter) {
            Brcb(scales[quarter * FLOATS_PER_BLOCK], tileScale_.Get<float>()[rowBase * blocks],
                static_cast<uint8_t>(rows * blocks / FLOATS_PER_BLOCK), {4, 32});
        }
        // Brcb、Mul、Cast 同在 PIPE_V 上按序执行，写后读依赖不需要屏障；只在 Brcb 之前
        // 保留一次屏障，隔离 Gather 对 index_ 的读取与这里对 index_ 的复用。
        Mul(value, value, scales, count);
        Cast(output, value, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    }

    // FP4→BF16：仅 Gather 查表 + 尺度相乘 + 一次 float→bf16 CAST_RINT，输出 2 字节/元素。
    // float→bf16 必须用 CAST_RINT（就近舍入），CAST_NONE 的 32→16 截断会产生垃圾值/NaN。
    __aicore__ inline void DecodeRowsBf16(const LocalTensor<uint8_t> &packed,
        uint32_t rowBase, LocalTensor<bfloat16_t> output, uint32_t tileK, uint32_t rows = DECODE_ROWS)
    {
        if (usePairLut_) {
            DecodeRowsPairLut(packed, rowBase, output, tileK, rows);
            return;
        }
        const uint32_t count = rows * tileK;
        auto value = floats_.Get<float>();
        auto halfCodes = half_.Get<half>();
        auto index = index_.Get<int32_t>();
        // 尺度广播缓冲 c 复用 index_ 的死区：Gather 完成后 index_ 不再被读取，c 在 Gather
        // 之后才写入（两者之间有 PipeBarrier 隔离），且二者字节数相同（DECODE_ELEMENTS×4）。
        // 省掉原来 floats_[DECODE_ELEMENTS] 的 32KB，使 BF16 cache 算子的 AIV UB 不超 192KB。
        auto c = index_.Get<float>();
        auto tileScale = tileScale_.Get<float>();

        Cast(halfCodes, packed.ReinterpretCast<int4b_t>(), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        static_assert(sizeof(float) == 4, "Gather LUT entries must be FP32");
        constexpr half FP4_CODE_BIAS = static_cast<half>(8.0f);
        constexpr half LUT_ENTRY_BYTES = static_cast<half>(4.0f);
        Adds(halfCodes, halfCodes, FP4_CODE_BIAS, count);
        PipeBarrier<PIPE_V>();
        Muls(halfCodes, halfCodes, LUT_ENTRY_BYTES, count);
        PipeBarrier<PIPE_V>();
        Cast(index, halfCodes, RoundMode::CAST_TRUNC, count);
        PipeBarrier<PIPE_V>();
        Gather(value, magTable_.Get<float>(), index.ReinterpretCast<uint32_t>(),
            static_cast<uint32_t>(0), count);
        PipeBarrier<PIPE_V>();

        const uint32_t blocks = tileK / SCALE_BLOCK;
        for (uint32_t quarter = 0; quarter < SCALE_BLOCK / FLOATS_PER_BLOCK; ++quarter) {
            Brcb(c[quarter * FLOATS_PER_BLOCK], tileScale[rowBase * blocks],
                static_cast<uint8_t>(rows * blocks / FLOATS_PER_BLOCK), {4, 32});
        }
        PipeBarrier<PIPE_V>();

        Mul(value, value, c, count);
        PipeBarrier<PIPE_V>();
        Cast(output, value, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    }

private:
    TBuf<QuePosition::VECCALC> magTable_, half_, index_, floats_, tileScale_, taskScales_, pairLut_;
    bool usePairLut_ = false;
};

// BF16 权重解码发布流水：与 INT8 路径共用同一套 MTE2 预取/tile 边界/队列 FIFO 结构，
// 仅输出类型换为 BF16（每元素 2 字节、GM 行跨步翻倍）。普通算子不写专家缓存（doWriteThrough=false）；
// 缓存算子需要填充缓存槽时才走写穿分支，把同一份已解码 BF16 同时写入 bf16_cache 槽位。
template <class TilingData>
__aicore__ inline void PublishDecodeBf16(
    uint32_t expert, uint32_t n, uint32_t tile, uint32_t wSlot,
    int32_t writeSlot, bool doWriteThrough, uint32_t sub, const TilingData &t,
    VectorDecoder &decoder,
    TQue<QuePosition::VECIN, STAGES> &packedQueue,
    TQue<QuePosition::VECOUT, STAGES> &decodedQueue,
    GlobalTensor<uint8_t> weight, GlobalTensor<bfloat16_t> scratchWeight,
    GlobalTensor<bfloat16_t> bf16Cache)
{
    const uint32_t halfN = TILE_N / 2;
    uint64_t weightSlotOffset = static_cast<uint64_t>(wSlot) * TILE_N * t.k;
    uint64_t kBase = static_cast<uint64_t>(tile) * t.tileK;
    const uint32_t batches = halfN / DECODE_ROWS;
    const uint32_t tiles = t.k / t.tileK;
    const bool hasNextTile = (tile + 1 < tiles);
    const uint64_t srcRowBase =
        (static_cast<uint64_t>(expert) * t.n + n + sub * halfN) * (t.k / 2) + kBase / 2;
    const uint64_t dstRowBase = weightSlotOffset + static_cast<uint64_t>(sub * halfN) * t.k + kBase;
    const uint32_t bf16RowStride = t.k * static_cast<uint32_t>(sizeof(bfloat16_t));

    if (tile == 0) {
        auto packed0 = packedQueue.AllocTensor<uint8_t>();
        decoder.LoadPackedRows(packed0, weight[srcRowBase], t.k / 2, t.tileK);
        packedQueue.EnQue(packed0);
    }

    for (uint32_t b = 0; b < batches; ++b) {
        const uint32_t row = b * DECODE_ROWS;

        if (b + 1 < batches || hasNextTile) {
            const uint64_t nextSrc = (b + 1 < batches)
                ? srcRowBase + static_cast<uint64_t>(row + DECODE_ROWS) * (t.k / 2)
                : srcRowBase + static_cast<uint64_t>(t.tileK) / 2;
            auto nextPacked = packedQueue.AllocTensor<uint8_t>();
            decoder.LoadPackedRows(nextPacked, weight[nextSrc], t.k / 2, t.tileK);
            packedQueue.EnQue(nextPacked);
        }

        auto packed = packedQueue.DeQue<uint8_t>();
        auto decoded = decodedQueue.AllocTensor<bfloat16_t>();
        decoder.DecodeRowsBf16(packed, row, decoded, t.tileK);
        packedQueue.FreeTensor(packed);

        decodedQueue.EnQue(decoded);
        decoded = decodedQueue.DeQue<bfloat16_t>();
        decoder.StoreDecodedRowsBf16(
            scratchWeight[dstRowBase + static_cast<uint64_t>(row) * t.k],
            decoded, bf16RowStride, t.tileK);
        if (doWriteThrough) {
            // 写穿（Write-Through）：同一份已解码 BF16 同时写入持久缓存槽位。
            // 先选持久缓存槽，每槽包含完整 N×K 个 BF16 权重。
            uint64_t cacheRowBase = static_cast<uint64_t>(writeSlot) * t.n * t.k
                // 再定位本次 N64 块、本 AIV 的通道批次及当前 K tile。
                + (static_cast<uint64_t>(n) + sub * halfN + row) * t.k + kBase;
            // 同一份 BF16 结果再写入持久缓存，供后续调用命中时复用。
            decoder.StoreDecodedRowsBf16(bf16Cache[cacheRowBase], decoded, bf16RowStride, t.tileK);
        }
        decodedQueue.FreeTensor(decoded);
    }
}
} // namespace Fp4Int8V2
#endif

// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_FP4_CACHE_LRU_H
#define VLLM_ASCEND_FP4_CACHE_LRU_H

#include "kernel_operator.h"

namespace Fp4Int8V2 {
using namespace AscendC;

// 热点专家缓存槽位上限。缓存容量（capacity）与专家数（experts）均受 tiling 校验
// 限制在 256 以内，因此 slot/resident（按专家索引）与 finalState/finalTimestamps
// （按槽位索引）共用同一上限。
constexpr uint32_t MAX_CACHE_SLOTS = 256;

// 仅在 ComputeCacheMap 内部标记低于缓存门限的专家；返回前恢复为公开的 -1。
constexpr int32_t INACTIVE_EXPERT_SLOT = -2;

// 确定性 LRU 槽位映射结果。
//
// 由每个 AIV 核冗余、一致地计算（读同一份 step-start 缓存状态，无全局屏障）：
//   - slot[e]        专家 e 本 step 使用的缓存槽位，或 -1（本 step 不缓存）
//   - resident[e]    1 命中（从缓存复制），0 未命中（解码）
//   - finalState[s]  槽位 s 的最终占用专家，或 -1 空
//   - finalTimestamps[s] 槽位 s 的最终时间戳
//
// 关键不变量（保证多核写回无竞争、命中读与未命中写不冲突）：
//   1. 命中专家占用的槽位在本 step 内被“钉住”，不被任何未命中专家淘汰（读不写竞态）；
//   2. 只有“最终占用者”（finalState[slot]==expert）才会回写 int8_cache，
//      被本 step 内后续未命中专家再次淘汰的专家只解码、不回写（写写竞态）；
//   3. 前 numShared 个槽位为共享专家永久槽位，从不参与淘汰。
struct ExpertSlotMap {
    int32_t slot[MAX_CACHE_SLOTS];
    uint8_t resident[MAX_CACHE_SLOTS];
    int32_t finalState[MAX_CACHE_SLOTS];
    int32_t finalTimestamps[MAX_CACHE_SLOTS];
};

// 专家 e 本 step 的路由 token 数：groupType==1 时 groups[e] 为原始计数；
// groupType==0 时 groups[e] 为前缀和（结束偏移），须减去上一项。
__aicore__ inline int64_t ExpertTokenCount(const GlobalTensor<int64_t> &groups,
    uint32_t groupType, uint32_t expert)
{
    int64_t value = groups.GetValue(expert);
    if (groupType == 0) {
        int64_t prev = expert == 0 ? 0 : groups.GetValue(expert - 1);
        return value - prev;
    }
    return value;
}

// 确定性、无竞争的 LRU 槽位映射（详见 fp4_cache_lru.h 顶部不变量）。
// 只读 cacheState/cacheTimestamps（step-start 快照），只写 map 输出；
// 调用方须将 cacheState/cacheTimestamps 指向双缓冲 [2, capacity] 的 active 行
// （本步读第 currentStep%2 行），最终状态的持久化由调用方写到 inactive 行
// （第 (currentStep+1)%2 行），且仅在单一 AIV 上执行一次以保证幂等。
__aicore__ inline void ComputeCacheMap(
    const GlobalTensor<int64_t> &groups, uint32_t experts, uint32_t groupType,
    const GlobalTensor<int32_t> &cacheState, const GlobalTensor<int32_t> &cacheTimestamps,
    uint32_t capacity, uint32_t hitThreshold, uint32_t currentStep, uint32_t numShared,
    ExpertSlotMap &map)
{
    for (uint32_t e = 0; e < experts; ++e) {
        map.resident[e] = 0;
        map.slot[e] = -1;
    }

    // 每个槽位只从 GM 读取一次，同时反向建立 expert -> step-start slot。
    // 复用现有输出数组，不增加 AIV 栈数组或 UB/GM 工作区。
    for (uint32_t s = 0; s < capacity; ++s) {
        int32_t occ = cacheState.GetValue(s);
        map.finalState[s] = occ;
        map.finalTimestamps[s] = cacheTimestamps.GetValue(s);
        if (occ >= 0 && static_cast<uint32_t>(occ) < experts && map.slot[occ] == -1) {
            // 若输入快照包含重复专家，保持原实现的最低索引命中槽位。
            map.slot[occ] = static_cast<int32_t>(s);
        }
    }

    // 第一遍：确定命中标志与命中槽位，以及共享专家的固定槽位。
    // 前缀和输入按专家顺序单次读取，保留前一项，避免 GetValue(e-1)。
    int64_t previousEnd = (groupType == 0 && numShared > 0 && numShared < experts)
        ? groups.GetValue(numShared - 1) : 0;
    for (uint32_t e = 0; e < experts; ++e) {
        if (e < numShared) {
            // 共享专家永久固定到槽位 e（前 numShared 个槽位，从不淘汰）。
            map.slot[e] = static_cast<int32_t>(e);
            map.resident[e] = (map.finalState[e] == static_cast<int32_t>(e)) ? 1 : 0;
            continue;
        }
        int64_t value = groups.GetValue(e);
        int64_t count = groupType == 0 ? value - previousEnd : value;
        previousEnd = value;
        if (count < static_cast<int64_t>(hitThreshold)) {
            map.slot[e] = INACTIVE_EXPERT_SLOT;
            continue; // 冷专家：本 step 不缓存
        }
        if (map.slot[e] >= 0) {
            map.resident[e] = 1;
        }
    }

    // 第二遍：命中刷新时间戳，共享未命中填充固定槽位，路由未命中按 LRU 淘汰。
    for (uint32_t e = 0; e < experts; ++e) {
        if (e < numShared) {
            if (!map.resident[e]) {
                map.finalState[e] = static_cast<int32_t>(e);
                map.finalTimestamps[e] = static_cast<int32_t>(currentStep);
            }
            continue;
        }
        if (map.slot[e] == INACTIVE_EXPERT_SLOT) {
            map.slot[e] = -1;
            continue; // 冷专家
        }
        if (map.resident[e]) {
            // 命中：仅刷新访问时间戳（touch），槽位保持“钉住”。
            map.finalTimestamps[static_cast<uint32_t>(map.slot[e])] = static_cast<int32_t>(currentStep);
            continue;
        }
        // 未命中：在 [numShared, capacity) 内淘汰。跳过被命中专家“钉住”的槽位；
        // 优先选第一个空槽，否则选时间戳最小的槽位。
        int32_t bestEmpty = -1;
        int32_t bestIdx = -1;
        int32_t bestTs = 0x7FFFFFFF;
        for (uint32_t s = numShared; s < capacity; ++s) {
            // 已命中的槽位始终被钉住，因此工作副本中不会改变其占用者；
            // 新写入的未命中专家 resident 始终为 0。可直接检查工作副本，
            // 无需在每个未命中专家的淘汰扫描中重复读取 GM 快照。
            int32_t occ = map.finalState[s];
            if (occ >= 0 && map.resident[static_cast<uint32_t>(occ)] == 1) {
                continue; // 钉住：step-start 占用者为本 step 命中专家，正在读取该槽位
            }
            if (map.finalState[s] == -1) {
                bestEmpty = static_cast<int32_t>(s);
                break; // 最低索引空槽优先
            }
            if (map.finalTimestamps[s] < bestTs) {
                bestTs = map.finalTimestamps[s];
                bestIdx = static_cast<int32_t>(s);
            }
        }
        int32_t slot = (bestEmpty != -1) ? bestEmpty : bestIdx;
        if (slot != -1) {
            map.slot[e] = slot;
            map.finalState[static_cast<uint32_t>(slot)] = static_cast<int32_t>(e);
            map.finalTimestamps[static_cast<uint32_t>(slot)] = static_cast<int32_t>(currentStep);
        }
        // slot == -1：无可用淘汰槽位（缓存满且全部命中），仅解码不回写。
    }
}

} // namespace Fp4Int8V2
#endif

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_fp4_tiling.cpp
 * \brief Tiling implementation for DispatchFFNCombineFP4
 */

#include <vector>
#include <map>
#include <algorithm>
#include <cstring>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/error_log.h"
#include "register/op_def_registry.h"
#include "dispatch_ffn_combine_fp4_tiling.h"
#include "moe_init_routing_v2/moe_init_routing_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
    const char *K_INNER_DEBUG = "DispatchFFNCombineFP4 Tiling Debug";
    constexpr uint32_t ATTR_GROUP_INDEX = 0;
    constexpr uint32_t ATTR_MAX_OUTPUT_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_IS_TRANS_B = 2;
    constexpr uint32_t ATTR_WEIGHT_NZ = 3;
    constexpr uint32_t ATTR_SWIGLU_LIMIT = 4;
    constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 5;
    constexpr int64_t MAX_WORLD_SIZE = 768;
    constexpr uint64_t INIT_TILINGKEY = 1000000;
    constexpr uint64_t TILINGKEY_TRANS_B = 1U;
    constexpr uint64_t TILINGKEY_WEIGHT_NZ = 10;
    constexpr uint32_t X_INDEX = 0;
    constexpr uint32_t WEIGHT_INDEX = 1;
    constexpr uint32_t WEIGHT2_INDEX = 2;
    constexpr uint32_t EXPERTID_INDEX = 3;
    constexpr uint32_t BLOCK_NUM = 20;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint64_t MB_SIZE = 1024 * 1024UL;
}

namespace optiling {

static ge::graphStatus DispatchFFNCombineFP4CheckAttrAndSetTiling(gert::TilingContext *context, DispatchFFNCombineFP4Info& info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto maxOutputSizePtr = attrs->GetAttrPointer<int>(ATTR_MAX_OUTPUT_SIZE_INDEX);
    auto is_trans_b = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B);
    auto weight_nz = attrs->GetAttrPointer<bool>(ATTR_WEIGHT_NZ);
    auto swiglu_limit = attrs->GetAttrPointer<float>(ATTR_SWIGLU_LIMIT);

    OP_TILING_CHECK(groupPtr == nullptr || strlen(groupPtr) == 0,
        OP_LOGE(K_INNER_DEBUG, "group is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(is_trans_b == nullptr,
        OP_LOGE(K_INNER_DEBUG, "is_trans_b is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(weight_nz == nullptr,
        OP_LOGE(K_INNER_DEBUG, "weight_nz is invalid."), return GRAPH_FAILED);

    info.maxOutputSize = maxOutputSizePtr != nullptr ? *maxOutputSizePtr : 0;
    info.isTransposeB = *is_trans_b;
    info.isWeightNz = *weight_nz;
    info.swigluLimit = swiglu_limit != nullptr ? *swiglu_limit : 0.0f;

    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_WORLD_SIZE_INDEX));
    OP_TILING_CHECK(worldSizePtr == nullptr || *worldSizePtr <= 0 || *worldSizePtr > MAX_WORLD_SIZE,
        OP_LOGE(K_INNER_DEBUG, "worldSize is invalid, expect (0, %lld], but is %lld.",
            static_cast<long long>(MAX_WORLD_SIZE),
            worldSizePtr == nullptr ? 0L : static_cast<long long>(*worldSizePtr)), return ge::GRAPH_FAILED);
    info.worldSize = static_cast<uint32_t>(*worldSizePtr);

    OP_LOGD(K_INNER_DEBUG, "maxOutputSize=%d, rankSize=%d, swigluLimit=%f",
            info.maxOutputSize, info.worldSize, info.swigluLimit);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineFP4CheckShapeAndSetTiling(gert::TilingContext *context, DispatchFFNCombineFP4Info &info)
{
    const gert::StorageShape *aStorageShape = context->GetInputShape(X_INDEX);
    auto expertIdxTensor = context->GetDynamicInputTensor(EXPERTID_INDEX, 0);
    uint32_t M = aStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = aStorageShape->GetStorageShape().GetDim(1);

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    uint32_t wTensorDims = wTensor->GetOriginShape().GetDimNum();
    uint32_t N;
    uint32_t expertPerRank;

    uint32_t listLen = 0;
    while (true) {
        auto wTensorT = context->GetDynamicInputTensor(WEIGHT_INDEX, ++listLen);
        if (wTensorT == nullptr) { break; }
    }

    if (wTensorDims == 3) {
        // [E, N, K/2]
        expertPerRank = wTensor->GetStorageShape().GetDim(0);
        N = wTensor->GetStorageShape().GetDim(1);
    } else {
        // [N, K/2] per expert
        expertPerRank = listLen;
        N = wTensor->GetStorageShape().GetDim(0);
    }

    uint32_t topK = expertIdxTensor->GetStorageShape().GetDim(1);

    info.M = M;
    info.N = N;
    info.K = K;
    info.expertPerRank = expertPerRank;
    info.topK = topK;
    info.listLen = listLen;
    info.usePairLut = (K == 4096 || K == 2048) ? 1 : 0;

    OP_LOGD(K_INNER_DEBUG, "M=%d, K=%d, N=%d, expertPerRank=%d, topK=%d, listLen=%d, usePairLut=%d",
            info.M, info.K, info.N, info.expertPerRank, info.topK, info.listLen, info.usePairLut);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineFP4GetPlatformInfoAndSetTiling(gert::TilingContext *context, DispatchFFNCombineFP4Info& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d, ubSize=%lu", info.aivNum, info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

void SetTilingData(CoCTiling &cocTilingData, DispatchFFNCombineFP4Info &info)
{
    cocTilingData.m0 = 128;
    cocTilingData.k0 = 256;
    cocTilingData.n0 = 256;
    cocTilingData.swizzleDirect = 1;
    cocTilingData.swizzleOffset = 7;
    cocTilingData.ubMoveNum = 16 * 1024;
    cocTilingData.pValue = 1;
    cocTilingData.commNpuSplit = info.worldSize;
    cocTilingData.commDataSplit = 1;
    cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 / 2;
}

static ge::graphStatus DispatchFFNCombineFP4TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter DispatchFFNCombineFP4 tiling func.");

    DispatchFFNCombineFP4TilingData *tilingData = context->GetTilingData<DispatchFFNCombineFP4TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);

    DispatchFFNCombineFP4Info& info = tilingData->dispatchFFNCombineFP4Info;

    OP_TILING_CHECK(DispatchFFNCombineFP4CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineFP4 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineFP4CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineFP4 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineFP4GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombineFP4 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);

    SetTilingData(tilingData->cocTiling, info);

    // Block dim
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Tiling key
    uint64_t tilingKey = INIT_TILINGKEY;
    tilingKey += info.isTransposeB ? TILINGKEY_TRANS_B : 0;
    tilingKey += info.isWeightNz ? TILINGKEY_WEIGHT_NZ : 0;
    context->SetTilingKey(tilingKey);

    // Moe init routing
    optiling::MoeInitRoutingV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = info.expertPerRank * info.worldSize;
    int64_t activeNum = info.M * info.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;
    uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
    moeInitRoutingQuantV2TilingBase.DoTiling(info.M, info.K, info.topK, expertCapacity, expertNum, activeNum, dropPadMode,
        expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantMode, scaleDim0, aivNumInitRouting, ubSize);

    tilingData->cocTiling.moeInitRoutingQuantV2TilingData = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.srcToDstComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp = moeInitRoutingQuantV2TilingBase.moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->cocTiling.initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;

    // Workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;
    uint64_t cocWorkspace = (info.M + 256 - 1) / 256 * 256 * info.topK * sizeof(int32_t) +
                            info.worldSize * info.worldSize * info.expertPerRank * sizeof(int32_t) * 3 +
                            info.maxOutputSize * sizeof(float) * 2 +
                            info.maxOutputSize * info.N * sizeof(int16_t) +
                            info.maxOutputSize * n2 * sizeof(int16_t) +
                            info.maxOutputSize * info.K * sizeof(int16_t) +
                            info.maxOutputSize * k2 * sizeof(int16_t) +
                            info.worldSize * sizeof(int32_t) * 16 +
                            (info.expertPerRank + info.worldSize) * sizeof(int32_t) * 16;

    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, static_cast<uint64_t>(initRoutingWorkspace)) + 32 * MB_SIZE;

    // Communication tiling
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    uint32_t opType = 8U;
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OP_LOGI(nodeName, "Leave DispatchFFNCombineFP4 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineFP4TilingFunc(gert::TilingContext* context)
{
    return DispatchFFNCombineFP4TilingFuncImpl(context);
}

struct DispatchFFNCombineFP4CompileInfo {};
ge::graphStatus TilingParseForDispatchFFNCombineFP4(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchFFNCombineFP4)
    .Tiling(DispatchFFNCombineFP4TilingFunc)
    .TilingParse<DispatchFFNCombineFP4CompileInfo>(TilingParseForDispatchFFNCombineFP4);
} // namespace optiling

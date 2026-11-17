/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_dispatch_ffn_combine_fp4.h"
#include <algorithm>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <climits>

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDispatchFFNCombineFP4GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensor* probs, const aclTensor* xActiveMask,
    const char* group, int64_t maxOutputSize,
    bool transB, bool weightNz, double swigluLimit, int64_t worldSize,
    const aclTensor* out, const aclTensor* expertTokenNums,
    uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDispatchFFNCombineFP4(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnDispatchFFNCombineFP4GetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensor* probs, const aclTensor* xActiveMask,
    const char* group, int64_t maxOutputSize, double swigluLimit, int64_t worldSize,
    const aclTensor* out, const aclTensor* expertTokenNums,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    bool transB = false;
    bool weightNz = true;

    aclnnStatus ret = aclnnInnerDispatchFFNCombineFP4GetWorkspaceSize(
        x, weight1, weight2, expertId,
        scale1, scale2,
        probs, xActiveMask,
        group, maxOutputSize,
        transB, weightNz, swigluLimit, worldSize,
        out, expertTokenNums, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnDispatchFFNCombineFP4(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerDispatchFFNCombineFP4(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif

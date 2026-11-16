# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(MAKESELF_NAME "makeself")
set(MAKESELF_PATH ${CANN_3RD_LIB_PATH}/makeself)

# 默认配置的makeself还是不存在则优先解压本地包，否则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    file(MAKE_DIRECTORY "${MAKESELF_PATH}")
    if (EXISTS "${CANN_3RD_LIB_PATH}/pkg/makeself-release-2.5.0-patch1.tar.gz")
        message(STATUS "Extracting local makeself from ${CANN_3RD_LIB_PATH}/pkg/makeself-release-2.5.0-patch1.tar.gz")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CANN_3RD_LIB_PATH}/pkg/makeself-release-2.5.0-patch1.tar.gz"
                        WORKING_DIRECTORY "${MAKESELF_PATH}")
    elseif (EXISTS "${PROJECT_SOURCE_DIR}/third_party/pkg/makeself-release-2.5.0-patch1.tar.gz")
        message(STATUS "Extracting local makeself from ${PROJECT_SOURCE_DIR}/third_party/pkg/makeself-release-2.5.0-patch1.tar.gz")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${PROJECT_SOURCE_DIR}/third_party/pkg/makeself-release-2.5.0-patch1.tar.gz"
                        WORKING_DIRECTORY "${MAKESELF_PATH}")
    else()
        set(MAKESELF_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
        message(STATUS "Downloading ${MAKESELF_NAME} from ${MAKESELF_URL}")

        include(FetchContent)
        FetchContent_Declare(
            ${MAKESELF_NAME}
            URL ${MAKESELF_URL}
            URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
            SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
        )
        FetchContent_MakeAvailable(${MAKESELF_NAME})
    endif()
endif()

if (EXISTS "${MAKESELF_PATH}/makeself.sh")
    execute_process(
        COMMAND chmod 755 "${MAKESELF_PATH}/makeself.sh"
        COMMAND chmod 755 "${MAKESELF_PATH}/makeself-header.sh"
        RESULT_VARIABLE CHMOD_RESULT
        ERROR_VARIABLE CHMOD_ERROR
    )
endif()
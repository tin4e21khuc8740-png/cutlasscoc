#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib> // For std::atoi
#include <cmath>   // For std::fabs
#include <algorithm>
#include <numeric>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v2.h"
#include "data_utils.h"
// #include "golden.hpp"

// int REPEATTIMES = 10;

bool ReadFileToVector(const std::string &filePath, std::vector<__fp16> &data)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(__fp16));
    file.close();
    return true;
}

bool ReadFileToVector(const std::string &filePath, std::vector<int32_t> &data)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(int32_t));
    file.close();
    return true;
}

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape)
    {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, bool isColumnMajor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    std::vector<T> hostData(size, 1);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    if (isColumnMajor)
    {
        for (int64_t i = 1; i < shape.size(); i++)
        {
            strides[i] = shape[i - 1] * strides[i - 1];
        }
    }
    else
    {
        for (int64_t i = shape.size() - 2; i >= 0; i--)
        {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 比较两个向量是否相等
bool CompareResults(const std::vector<__fp16> &result, const std::vector<__fp16> &expected, int M, int K, int N, float tolerance = 1e-6f)
{
    if (result.size() != expected.size())
    {
        std::cerr << "Size mismatch between result and expected data!" << std::endl;
        return false;
    }

    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
    float sum_absolute_error = 0.0f;
    float sum_relative_error = 0.0f;
    int count = 0;

    for (size_t i = 0; i < result.size(); ++i)
    {
        float actual = static_cast<float>(result[i]);
        float expected_value = static_cast<float>(expected[i]);
        float diff = std::fabs(actual - expected_value);
        float relative_error = diff / (std::fabs(expected_value) + 1e-7);

        if (diff > max_absolute_error)
        {
            max_absolute_error = diff;
        }

        if (relative_error > max_relative_error)
        {
            max_relative_error = relative_error;
        }

        sum_absolute_error += diff;
        sum_relative_error += relative_error;

        if (diff > tolerance && count < 127)
        {
            std::cerr << "Mismatch at index " << i << ": result=" << actual
                      << ", expected=" << expected_value << ", diff=" << diff << std::endl;
        }

        if (relative_error > tolerance)
        {
            count++;
        }
    }
    float avg_absolute_error = sum_absolute_error / result.size();
    float avg_relative_error = sum_relative_error / result.size();
    float relative_error_ratio = static_cast<float>(count) / result.size();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix dimensions: M = " << M << ", K = " << K << ", N = " << N << ", max_absolute_error: " << max_absolute_error << ", max_relative_error: " << max_relative_error << ", avg_absolute_error: " << avg_absolute_error << ", avg_relative_error: " << avg_relative_error << ", relative_error_ratio: " << relative_error_ratio << std::endl;

    return true;
}

int CreateAclTensorList(const std::vector<std::vector<int64_t>> &shapes, void **deviceAddr,
                        aclDataType dataType, aclTensorList **tensor, bool isColumnMajor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++)
    {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i, isColumnMajor);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

std::vector<int32_t> GenerateRandomGlobalTokensPerExpert(int epSize, int expertNum, int tokenNum, bool isPrint,
                                                         int seed = 0)
{
    std::vector<int32_t> globalTokensPerExpertData;
    globalTokensPerExpertData.reserve(epSize * expertNum);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int32_t> distribution(0, tokenNum);
    std::vector<int32_t> tokensAccum(expertNum + 1);
    tokensAccum[0] = 0;
    tokensAccum[expertNum] = tokenNum;
    for (int epIdx = 0; epIdx < epSize; ++epIdx)
    {
        for (int i = 1; i < expertNum; ++i)
        {
            tokensAccum[i] = distribution(gen);
        }
        std::sort(tokensAccum.begin() + 1, tokensAccum.begin() + expertNum);
        for (int i = 0; i < expertNum; ++i)
        {
            globalTokensPerExpertData.push_back(tokensAccum[i + 1] - tokensAccum[i]);
        }
    }

    if (isPrint)
    {
        // 打印随机生成的 global tokens 表
        std::cout << "Global tokens per expert:\n";
        for (int epIdx = 0; epIdx < epSize; ++epIdx)
        {
            int epExpertStart = epIdx * expertNum;
            std::cout << globalTokensPerExpertData[epExpertStart];
            for (int expertIdx = 1; expertIdx < expertNum; ++expertIdx)
            {
                std::cout << " " << globalTokensPerExpertData[epExpertStart + expertIdx];
            }
            std::cout << "\n";
        }
    }

    return globalTokensPerExpertData;
}

int main(int argc, char *argv[])
{
    int64_t M = std::atoi(argv[1]);
    int64_t N = std::atoi(argv[2]);
    int64_t K = std::atoi(argv[3]);
    int rank = std::atoi(argv[4]);
    std::string mode = argv[5];
    int rankSize = std::atoi(argv[6]);
    int groupSize = std::atoi(argv[7]);
    int transB = std::atoi(argv[8]);
    int start_id = std::atoi(argv[9]);

    aclrtStream stream;
    auto ret = Init(start_id + rank, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int32_t> globalTokensPerExpert = GenerateRandomGlobalTokensPerExpert(rankSize, rankSize * groupSize, M, false, 0);
    if (rank == start_id)
    {
        std::cout << "globalTokensPerExpert data in aclnn:" << std::endl;
        for (int i = 0; i < rankSize; i++)
        {
            for (int j = 0; j < groupSize * rankSize; j++)
            {
                std::cout << globalTokensPerExpert[i * groupSize * rankSize + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    int realGroupSize = 0;
    std::vector<std::vector<int64_t>> xShape;
    std::vector<std::vector<int64_t>> weightShape;
    std::vector<std::vector<int64_t>> yShape;
    std::vector<std::vector<int64_t>> biasShape;
    for (int i = 0; i < groupSize; i++)
    {
        int globalExpertIdx = rank * groupSize + i;
        int64_t temp = 0;
        for (int j = 0; j < rankSize; j++)
        {
            temp += globalTokensPerExpert[j * rankSize * groupSize + globalExpertIdx];
        }
        if (temp > 0)
        {
            xShape.push_back({temp, K});
            weightShape.push_back({K, N});
            yShape.push_back({temp, N});
            biasShape.push_back({N});
            realGroupSize++;
        }
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    void *xDeviceAddr[realGroupSize];
    void *weightDeviceAddr[realGroupSize];
    void *biasDeviceAddr[realGroupSize];
    void *yDeviceAddr[realGroupSize];
    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *bias = nullptr;
    aclIntArray *groupedList = nullptr;
    aclTensorList *scale = nullptr;
    aclTensorList *offset = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *antiquantOffset = nullptr;
    aclTensorList *y = nullptr;
    int64_t splitItem = 0;
    int64_t groupType = -1;

    // 创建x aclTensorList
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x, false);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensorList
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight, transB);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensorList
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias, false);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensorList
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y, false);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmul第一段接口
    void *workspaceAddr = nullptr;
    // for (int i = 0; i < REPEATTIMES; i++)
    // {
        ret = aclnnGroupedMatmulV2GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, groupType, y, &workspaceSize, &executor);
        // std::cout << aclGetRecentErrMsg();
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0)
        {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // 调用aclnnGroupedMatmul第二段接口

        ret = aclnnGroupedMatmulV2(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // }

    // int errorPrintLimit = 16;
    // int errorCount = 0;
    // // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    // for (int i = 0; i < realGroupSize; i++) {
    //     auto size = GetShapeSize(yShape[i]);
    //     std::vector<uint16_t> resultData(size, 0);
    //     ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
    //                     size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    //     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    //     for (int64_t j = 0; j < size; j++) {
    //         if(resultData[j] != K){
    //             if(errorCount < errorPrintLimit){
    //                 std::cout << "rank " << rank << " group " << i << " y data index " << j << " result " << resultData[j] << " expected " << K << std::endl;
    //             }
    //             errorCount++;
    //         }
    //     }
    // }

    std::cout << "aclnnGroupedMatmul on rank " << rank << " finished.\n"
              << std::endl;

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    for (int i = 0; i < realGroupSize; i++)
    {
        aclrtFree(xDeviceAddr[i]);
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(yDeviceAddr[i]);
        aclrtFree(biasDeviceAddr[i]);
    }

    if (workspaceSize > 0)
    {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(start_id + rank);
    aclFinalize();
    return 0;
}

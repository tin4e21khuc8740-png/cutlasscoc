#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib> // For std::atoi
#include <cmath>   // For std::fabs
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"
#include "data_utils.h"
#include "golden.hpp"

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

#define CHECK_RET(cond, return_expr) \
  do                                 \
  {                                  \
    if (!(cond))                     \
    {                                \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do                                \
  {                                 \
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
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, bool isColumnMajor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

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

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main(int argc, char *argv[])
{
  // if (argc < 5) {
  //   std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <deviceId>" << std::endl;
  //   return -1;
  // }

  int M = std::atoi(argv[1]);
  int K = std::atoi(argv[2]);
  int N = std::atoi(argv[3]);
  int transA = std::atoi(argv[4]);
  int transB = std::atoi(argv[5]);
  int deviceId = std::atoi(argv[6]);
  std::string mode = argv[7];

  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  for(int i = 0; i < 10; i++){
    std::vector<int64_t> A_shape = {M, K};
    std::vector<int64_t> B_shape = {K, N};
    std::vector<int64_t> C_shape = {M, N};
    void *d_A = nullptr;
    void *d_B = nullptr;
    void *d_C = nullptr;
    aclTensor *tensor_A = nullptr;
    aclTensor *tensor_B = nullptr;
    aclTensor *tensor_C = nullptr;

    std::vector<__fp16> data_A(M * K);
    std::vector<__fp16> data_B(K * N);
    std::vector<__fp16> data_C(M * N);
    if (mode == "error")
    {
      golden::FillRandomData<__fp16>(data_A, -5.0f, 5.0f);
      golden::FillRandomData<__fp16>(data_B, -5.0f, 5.0f);
      golden::FillRandomData<__fp16>(data_C, -5.0f, 5.0f);
    }

    if (transA)
    {
      ret = CreateAclTensor(data_A, A_shape, &d_A, aclDataType::ACL_FLOAT16, &tensor_A, true);
    }
    else
    {
      ret = CreateAclTensor(data_A, A_shape, &d_A, aclDataType::ACL_FLOAT16, &tensor_A, false);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    if (transB)
    {
      ret = CreateAclTensor(data_B, B_shape, &d_B, aclDataType::ACL_FLOAT16, &tensor_B, true);
    }
    else
    {
      ret = CreateAclTensor(data_B, B_shape, &d_B, aclDataType::ACL_FLOAT16, &tensor_B, false);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(data_C, C_shape, &d_C, aclDataType::ACL_FLOAT16, &tensor_C, false);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int8_t cubeMathType = 2;
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;
    
    aclOpExecutor *executor;
    ret = aclnnMatmulGetWorkspaceSize(tensor_A, tensor_B, tensor_C, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0)
    {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclrtSynchronizeStream(stream);
    aclrtMemset(d_B, K * N, 0x1, K * N);

    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(C_shape);
    std::vector<__fp16> data_expected(size, static_cast<__fp16>(0));
    ret = aclrtMemcpy(data_expected.data(), data_expected.size() * sizeof(data_expected[0]), d_C,
                      size * sizeof(data_expected[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 比较expected 和 result
    if (mode == "error")
    {
      // 读取result 算子计算结果
      std::vector<__fp16> data_result(M * N);
      golden::ComputeMatmul(data_A, data_B, data_result, M, N, K);

      CompareResults(data_result, data_expected, M, K, N);
    }

    aclDestroyTensor(tensor_A);
    aclDestroyTensor(tensor_B);
    aclDestroyTensor(tensor_C);
    aclrtFree(d_A);
    aclrtFree(d_B);
    aclrtFree(d_C);
    if (workspaceSize > 0)
    {
      aclrtFree(workspaceAddr);
    }
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

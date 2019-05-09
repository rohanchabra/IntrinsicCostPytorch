#pragma once
#include <cuda_runtime.h>
constexpr int MAX_K = 35;

#define CUDA_ASSERT(e) { cudaAssert((e), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
          exit(code);
   }
}

int RoundDim(int N, int K)
{
  return (N + K - 1)/K;// + (N% K== 0 ? 0 : 1);
}

void GetKernelDims(dim3& gridDims,dim3& blockDims, int width,int height, const int BLOCK_SIZE = 16)
{
    blockDims = dim3(BLOCK_SIZE, BLOCK_SIZE);
    gridDims = dim3((width / blockDims.x) + (width  % blockDims.x == 0 ? 0 : 1),
     (height / blockDims.y) + (height% blockDims.y == 0 ? 0 : 1));
}

template<typename T>
__device__ bool InBounds(const T& valA,const T max,const T min = 0)
{
  return valA >= min && valA < max;
}

#include <cuda.h>
#include <cuda_runtime.h>

#include "cost_compute.h"
#include "CUDA/CostFuncs.cuh"
#include "CUDA/CostCompute.cpp"

void Intrinsic_Cost(at::Tensor leftFeat, at::Tensor rightFeat, at::Tensor costFeat, int windowSize)
{
  if(leftFeat.type().is_cuda())
    Cuda_Intrinsic_Cost<float>(leftFeat, rightFeat, costFeat, windowSize);

}

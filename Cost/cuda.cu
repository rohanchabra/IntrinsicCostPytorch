#include <ATen/ATen.h>
#include "CUDA/CostCompute.cu"

template void Cu_Intrinsic_Cost(float* leftFeat, float* rightFeat, float* cost, SAD_CostFunc costFunc, int width, int height, int dispMax, int numFeats, int windowSize);
template void Cu_Intrinsic_Cost(int* leftFeat, int* rightFeat, int* cost, SAD_CostFunc costFunc, int width, int height, int dispMax, int numFeats, int windowSize);

template void Cu_Intrinsic_Cost(float* leftFeat, float* rightFeat, float* cost, SSD_CostFunc costFunc, int width, int height, int dispMax, int numFeats, int windowSize);
template void Cu_Intrinsic_Cost(int* leftFeat, int* rightFeat, int* cost, SSD_CostFunc costFunc, int width, int height, int dispMax, int numFeats, int windowSize);

#pragma once

#include <cuda_runtime.h>

enum CostFuncType
{
  SSD_CF,
  SAD_CF
};

struct SSD_CostFunc
{
    template<typename T>
    __host__ __device__ T cost(const T& left,const T& right)
    {
      T diff = (left - right);
      return diff*diff;
    }
};



struct SAD_CostFunc
{
    template<typename T>
    __host__ __device__ T cost(const T& left,const T& right)
    {
      T diff = (left - right);
      return diff > 0 ? diff : - diff;
    }
};

#include "Utils.cuh"
#include "CostFuncs.cuh"
constexpr int MAX_DISP = 256;
template <typename T, typename CostFunc, int R >
__global__ void Kern_Intrinsic_Cost(T* leftFeat, T* rightFeat, T* costFeat, CostFunc costFunc,
                                    int width, int height, int dispMax, int numFeat)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  const int STEPS = 3;
  const int K = 16;

  int sidx = threadIdx.x + R;
  int sidy = threadIdx.y + R;

  T imLeft,imRight;

  T cost;

  __shared__ T  diff[K + 2*R][K + 2*R];

  // store needed values for left image into registers (constant indexed local vars)
  T imLeftA[STEPS];
  T imLeftB[STEPS];
  T finalCost[MAX_DISP];
  int F = numFeat;
  int validPix = 0;
  if(InBounds(u,width) && InBounds(v,height))
   validPix = 1;
  for(int f = 0; f < F; ++f)
  {
    #pragma unroll
    for (int i=0; i<STEPS; i++)
    {
        int offset = -R + i*R;
        int y = width*(v+offset)*F;
        if(InBounds(u-R,width) && InBounds(v+offset,height))
          imLeftA[i] = leftFeat[y + (u - R)*F + f];// tex2D(tex2Dleft, u-R, v+offset);
        else
          imLeftA[i] = 0;
        if(InBounds(u - R - K, width) && InBounds(v+offset,height))
          imLeftB[i] = leftFeat[y + (u - R + K)*F + f];//tex2D(tex2Dleft, u-R+blockDim.x, v+offset);
        else
          imLeftB[i] = 0;
    }

    int costIndex = v*width*dispMax*F + u*dispMax*F;

    for (int d = 0; d < dispMax; ++d)
    {
        //LEFT
        #pragma unroll
        for (int i=0; i<STEPS; i++)
        {
            int offset = -R + i *R;
            imLeft = imLeftA[i];
            if(InBounds(u - R - d, width) && InBounds(v+offset,height))
            {
              imRight = rightFeat[(v+offset)*width*F + (u - R - d)*F + f];//tex2D(tex2Dright, u-R+d, v+offset);
              cost = costFunc.cost(imLeft, imRight);
            }
            else
              cost = imLeft;
            diff[sidy+offset][sidx-R] = cost;
        }

        //RIGHT
        #pragma unroll
        for (int i=0; i<STEPS; i++)
        {
            int offset = -R + i*R;

            if (threadIdx.x < 2*R)
            {
                imLeft = imLeftB[i];
                if(InBounds(u - R - d + K, width) && InBounds(v+offset,height))
                {
                  imRight = rightFeat[(v+offset)*width*F + (u - R - d + K)*F + f];//tex2D(tex2Dright, u-R+K-d, v+offset);
                  cost = costFunc.cost(imLeft, imRight);
                }
                else
                  cost = imLeft;
                diff[sidy+offset][sidx-R+K] = cost;
            }
        }

        __syncthreads();

        // sum cost horizontally
  #pragma unroll
        for (int j=0; j<STEPS; j++)
        {
            int offset = -R + j*R;
            cost = 0;
  #pragma unroll

            for (int i=-R; i<=R ; i++)
            {
                cost += diff[sidy+offset][sidx+i];
            }

            __syncthreads();
            diff[sidy+offset][sidx] = cost;
            __syncthreads();

        }

        // sum cost vertically
        cost = 0;
  #pragma unroll

        for (int i=-R; i<=R ; i++)
        {
            cost += diff[sidy+i][sidx];
        }


        if(validPix)
          finalCost[d] = cost;//costFeat[costIndex + d*F +f] = cost;

        __syncthreads();

    }
    if(validPix)
    {
      for (int d = 0; d < dispMax; ++d)
        costFeat[costIndex + d*F +f] = finalCost[d];
    }
    __syncthreads();
  }

}

template <typename T, typename CostFunc >
void Cu_Intrinsic_Cost(T* leftFeat, T* rightFeat, T* costFeat, CostFunc costFunc, int width,
                      int height, int dispMax, int numFeats, int windowSize)
{
  const int K = 16;
  dim3 blockDims = dim3(K,K);
  dim3 gridDims = dim3(RoundDim(width,K),RoundDim(height,K));
  const int R = windowSize/2;
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  CUDA_ASSERT(cudaEventCreate(&start));
  CUDA_ASSERT(cudaEventCreate(&stop));

  // Record the start event
  CUDA_ASSERT(cudaEventRecord(start, NULL));
  if(R==1)
    Kern_Intrinsic_Cost<T,CostFunc,1><<<gridDims,blockDims>>>(leftFeat, rightFeat,
                                                              costFeat, costFunc, width, height, dispMax, numFeats);
  else if(R==2)
    Kern_Intrinsic_Cost<T,CostFunc,2><<<gridDims,blockDims>>>(leftFeat, rightFeat,
                                                              costFeat, costFunc, width, height, dispMax, numFeats);
  else if(R==3)
    Kern_Intrinsic_Cost<T,CostFunc,3><<<gridDims,blockDims>>>(leftFeat, rightFeat,
                                                              costFeat, costFunc, width, height, dispMax, numFeats);
  else if(R==4)
    Kern_Intrinsic_Cost<T,CostFunc,4><<<gridDims,blockDims>>>(leftFeat, rightFeat,
                                                              costFeat, costFunc, width, height, dispMax, numFeats);
  else if(R==5)
    Kern_Intrinsic_Cost<T,CostFunc,5><<<gridDims,blockDims>>>(leftFeat, rightFeat,
                                                              costFeat, costFunc, width, height, dispMax, numFeats);

  CUDA_ASSERT(cudaEventRecord(stop, NULL));
// Wait for the stop event to complete
  CUDA_ASSERT(cudaEventSynchronize(stop));


  float msecTotal = 0.0f;
  CUDA_ASSERT(cudaEventElapsedTime(&msecTotal, start, stop));
  //printf("Kernel processing time : %.4f (ms)\n", msecTotal);
  CUDA_ASSERT(cudaPeekAtLastError());

}

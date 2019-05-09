
SSD_CostFunc ssd;
SAD_CostFunc sad;

template <typename T, typename CostFunc>
void Cu_Intrinsic_Cost(T* leftFeat, T* rightFeat, T* costFeat, CostFunc costFunc, int width,
                      int height, int dispMax, int numFeats, int windowSize);


template <typename T>
void Cuda_Intrinsic_Cost(at::Tensor leftFeat, at::Tensor rightFeat, at::Tensor costFeat, int windowSize)
{
  int H = costFeat.size(0);
  int W = costFeat.size(1);
  int D = costFeat.size(2);
  int F = costFeat.size(3);

  auto lI = leftFeat.data<T>();
  auto rI = rightFeat.data<T>();
  auto costI = costFeat.data<T>();
  Cu_Intrinsic_Cost<T, SAD_CostFunc>(lI, rI, costI, sad, W, H, D, F, windowSize);
}

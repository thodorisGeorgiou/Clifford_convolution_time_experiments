#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "weight_to_angle_gradients.h"

// Define the CUDA kernel.

template <typename K>
__device__ void cudaCopyTo(const K *inpt, K *out, size_t size){
    for (size_t i = 0; i < size; i++) out[i] = inpt[i];
}

template <typename T>
__global__ void parallelCopyTo(const T *inpt, T *out, size_t size){
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x*gridDim.x;
    for (int i = threadID; i < size; i+=num_threads) out[i] = inpt[i];
}

template <typename T>
__global__ void padCudaKernel(T * tInput, const T* in, size_t depth, size_t bStep, size_t rStep, size_t obStep, size_t orStep, size_t oNumRows, size_t paddingOffsetRows, size_t paddingOffsetCols){
    size_t b = (size_t)blockIdx.x;
    size_t r = (size_t)threadIdx.x;
    T * tbp = tInput + b*bStep + r*rStep;
    if(r < paddingOffsetRows || r >= paddingOffsetRows + oNumRows)
      for(T *c = tbp; c<tbp+rStep;) *c++ = 0;
    else{
      T *tp = tbp + paddingOffsetCols*depth;
      const T *op = in + b*obStep + (r - paddingOffsetRows)*orStep;
      cudaCopyTo(op, tp, orStep);
      for(T *c=tbp; c < tp;) *c++ = 0;
      for(T *c=tp+orStep; c < tbp+rStep;) *c++ = 0;
    }
}

template <typename T>
__device__ void singlePointGatherAnglePositionGradients(T *padIn, T *output, const T gradient, T *woa, int inrStep, const int in_depth, \
  const int out_depth, const int filter_rows, const int filter_cols, int fcStep, int frStep){
    T result = 0;
    for(int r=0; r<filter_rows; r++){
      int frInd = r*frStep;
      float* inpt = padIn+(size_t)(r*inrStep);
      for(int c=0; c<filter_cols; c++){
        int fcInd = frInd + c*fcStep;
        for(int din=0; din<in_depth;din++){
          int fInd = fcInd + din*out_depth;
          result += *(inpt++) * woa[fInd];
        }
      }
    }
    *output = result * gradient;
}

template <typename T>
__device__ void singlePointGatherAngleRotationGradients(T *padIn, T *output, const T gradient, const T *weights, int inrStep, const int in_depth, \
  const int out_depth, const int filter_rows, const int filter_cols, int fcStep, int frStep){
    T result = 0;
    for(int r=0; r<filter_rows; r++){
      int frInd = r*frStep;
      float* inpt = padIn+(size_t)(r*inrStep);
      for(int c=0; c<filter_cols; c++){
        int fcInd = frInd + c*fcStep;
        for(int din=0; din<in_depth;din+=2){
          int fInd = fcInd + din*out_depth;
          result -= *(inpt++) * weights[fInd+out_depth];
          result += *(inpt++) * weights[fInd];
        }
      }
    }
    *output += gradient * result;
}


template <typename T>
__global__ void CalculateWeightToAngleGradientsCudaKernel(bool position, int index, T *padIn, T *output, const T *gradients, T *woa, const T *weights, const int *indexes, int outbStep, int outrStep, const int out_depth, \
  int inbStep, int inrStep, const int in_depth, const int filter_rows, const int filter_cols, int fcStep, int frStep, int weightSetSize, int numJobs){
      int threadID = blockIdx.x * blockDim.x + threadIdx.x;
      int num_threads = blockDim.x*gridDim.x;
      size_t weightSet = (size_t)(weightSetSize*index);
      for(int outIndex=threadID; outIndex<numJobs; outIndex+=num_threads){
        if(index != indexes[outIndex]) continue;
        int b = outIndex/outbStep;
        int rest = outIndex%outbStep;
        int r = rest / outrStep;
        int c = (rest % outrStep) / out_depth;
        int dout = (rest % outrStep) % out_depth;
        int inIndex = b*inbStep + r*inrStep + c*in_depth;
        size_t weightIndex = weightSet + (size_t)dout;
        singlePointGatherAnglePositionGradients(padIn+(size_t)inIndex, output+(size_t)outIndex, gradients[outIndex], woa+weightIndex, inrStep, in_depth,\
          out_depth, filter_rows, filter_cols, fcStep, frStep);
      }
}


        // if(position) singlePointGatherAnglePositionGradients(padIn+(size_t)inIndex, output+(size_t)outIndex, gradients[outIndex], woa+weightIndex, inrStep, in_depth,\
        //   out_depth, filter_rows, filter_cols, fcStep, frStep);
        // else singlePointGatherAngleRotationGradients(padIn+(size_t)inIndex, output+(size_t)outIndex, gradients[outIndex], weights+weightIndex, inrStep, in_depth,\
        //   out_depth, filter_rows, filter_cols, fcStep, frStep);

template <typename T>
__global__ void calculateWeightOverFiPos(T * woa, const T *weights, int weightSetSize, const int num_angles, float normFactor){
      int threadID = blockIdx.x * blockDim.x + threadIdx.x;
      int num_threads = blockDim.x*gridDim.x;
      // int wIndex = threadID;
      int wIndex = threadID+weightSetSize;
      for(; threadID<weightSetSize*(num_angles-1); threadID+=num_threads){
          woa[threadID] = weights[wIndex]/normFactor;
          wIndex += num_threads;
      }
}

template <typename T>
__global__ void calculateWeightOverFiNeg(T * woa, const T *weights, int weightSetSize, const int num_angles, float normFactor){
      int threadID = (blockIdx.x * blockDim.x + threadIdx.x) + weightSetSize;
      int num_threads = blockDim.x*gridDim.x;
      int wIndex = threadID - weightSetSize;
      for(; threadID<weightSetSize*(num_angles-1); threadID+=num_threads){
          woa[threadID] -= weights[wIndex]/normFactor;
          wIndex += num_threads;
      }
}

template <typename T>
__global__ void calculateWeightOverFiNegZero(T * woa, const T *weights, int weightSetSize, const int num_angles, float normFactor){
      int threadID = blockIdx.x * blockDim.x + threadIdx.x;
      int num_threads = blockDim.x*gridDim.x;
      int wIndex = threadID + (num_angles-2)*weightSetSize;
      for(; threadID<weightSetSize; threadID+=num_threads){
          woa[threadID] -= weights[wIndex]/normFactor;
          wIndex += num_threads;
      }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct CalculateWeightToAngleGradientsFunctor<GPUDevice, T> {
  void operator()(OpKernelContext *context, const Tensor& input, T* output, const int* indexes, const T *weights, const T* gradients, T *woa, \
    const int filter_rows, const int filter_cols, const int num_angles, const int in_depth, const int out_depth, \
    const int batch, const int rows, const int cols, Padding padding_, TensorFormat data_format_) {

  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  const T *inpt = input.flat<T>().data();

  int fcStep = out_depth*in_depth;
  int frStep = filter_cols*fcStep;
  int weightSetSize = filter_rows*frStep;
  float normFactor = M_PI*4/(float)(num_angles-1);
  // float normFactor = 2;
  // float normFactor = 0.0000001;

  CudaLaunchConfig cfg = GetCudaLaunchConfig((num_angles-1)*weightSetSize, context->eigen_device<GPUDevice>());
  calculateWeightOverFiPos<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (woa, weights, weightSetSize, num_angles, normFactor);
  cfg = GetCudaLaunchConfig((num_angles-2)*weightSetSize, context->eigen_device<GPUDevice>());
  cudaDeviceSynchronize();
  calculateWeightOverFiNeg<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (woa, weights, weightSetSize, num_angles, normFactor);
  cfg = GetCudaLaunchConfig(weightSetSize, context->eigen_device<GPUDevice>());
  cudaDeviceSynchronize();
  calculateWeightOverFiNegZero<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (woa, weights, weightSetSize, num_angles, normFactor);

  cudaDeviceSynchronize();
  // cfg = GetCudaLaunchConfig(weightSetSize, context->eigen_device<GPUDevice>());
  parallelCopyTo<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (woa, woa+(size_t)((num_angles-1)*weightSetSize), weightSetSize);

  int outrStep = out_depth*cols;
  int outbStep = rows*outrStep;

  int padding_rows = 0;
  int padding_cols = 0;
  Tensor paddedInput = input;
  int new_in_rows = rows;
  int new_in_cols = cols;
  int paddingOffsetRows = 0;
  int paddingOffsetCols = 0;

  if (padding_ == SAME) {
      padding_rows = std::max<int>(0, (rows - 1) + filter_rows - rows);
      padding_cols = std::max<int>(0, (cols - 1) + filter_cols - cols);
      Tensor transformed_input;
      new_in_rows = rows + padding_rows;
      new_in_cols = cols + padding_cols;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                  ShapeFromFormat(data_format_, batch, new_in_rows,
                  new_in_cols, in_depth), &transformed_input));

      paddingOffsetRows = padding_rows/2;
      paddingOffsetCols = padding_cols/2;
      T *tInput = transformed_input.flat<T>().data();
      int rStep = new_in_cols*in_depth;
      int bStep = rStep*new_in_rows;

      int orStep = cols*in_depth;
      int obStep = orStep*rows;
      /*Set padding rows and cols*/
      padCudaKernel<T><<<batch, new_in_rows, 0, context->eigen_device<GPUDevice>().stream()>>>
      (tInput, inpt, (size_t)in_depth, (size_t)bStep, (size_t)rStep, (size_t)obStep, (size_t)orStep, (size_t)rows, (size_t)paddingOffsetRows, (size_t)paddingOffsetCols);
      paddedInput = transformed_input;
  } /*Padding done*/

  T *padIn = paddedInput.flat<T>().data();
  int inrStep = in_depth*new_in_cols;
  int inbStep = inrStep*new_in_rows;

  cfg = GetCudaLaunchConfig(outbStep*batch, context->eigen_device<GPUDevice>());
  for(int index=0; index<num_angles; index++){
    cudaDeviceSynchronize();
    CalculateWeightToAngleGradientsCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
    (true, index, padIn, output, gradients, woa, weights, indexes, outbStep, outrStep, out_depth, inbStep, inrStep, in_depth,\
    filter_rows, filter_cols, fcStep, frStep, weightSetSize, outbStep*batch);
  }
  // for(int index=0; index<num_angles; index++){
  //   cudaDeviceSynchronize();
  //   CalculateWeightToAngleGradientsCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  //   (false, index, padIn, output, gradients, woa, weights, indexes, outbStep, outrStep, out_depth, inbStep, inrStep, in_depth,\
  //   filter_rows, filter_cols, fcStep, frStep, weightSetSize, outbStep*batch);
  // }
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CalculateWeightToAngleGradientsFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
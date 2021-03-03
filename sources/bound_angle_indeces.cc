#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"
// #include "tensorflow/core/framework/resource_mgr.h"
// #include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/use_cudnn.h"
// #include "tensorflow/core/kernels/conv_ops.h"
// #include "tensorflow/core/kernels/conv_2d.h"
// #include "tensorflow/core/util/tensor_format.h"

#include <iostream>
#include <omp.h>

using namespace tensorflow;

REGISTER_OP("BoundAngleIndeces")
    .Input("input: T")
    .Input("num_bins: T")
    .Output("output: T")
    // .Output("indeces: int32")
    .Attr("T: {int32}")
    // .Attr("T: {half, bfloat16, float, double}")
    .Attr("use_cudnn_on_gpu: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "bound_angle_indeces.h"


// CPU specialization of actual computation.
template <typename T>
struct BoundAngleIndecesFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* input, T* out, const int *numBins, const size_t dataLength) {
    int nb = *numBins;
    #pragma omp parallel for
    for(size_t p = 0; p<dataLength; p++){
        if(input[p]<0) out[p] = input[p] + nb;
        else if(input[p]>nb) out[p] = input[p] - nb;
        else out[p] = input[p];
    }
  }
};


template <typename Device, typename T>
class BoundAngleIndecesOp : public OpKernel {
 public:
  explicit BoundAngleIndecesOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }


  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);

    const Tensor& numBins = context->input(1);


    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("Input must have rank 4",
                                        input.shape().DebugString()));


    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, numBins.dims() == 1,
                errors::InvalidArgument("num bins should have rank 1",
                                        numBins.shape().DebugString()));

      OP_REQUIRES(
          context,
          numBins.dim_size(0) == 1,
          errors::InvalidArgument("num bins must be a single value"));


    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = input.dim_size(1);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = input.dim_size(2);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);

    // The first dimension for input is batch.
    const int64 batch_raw = input.dim_size(0);
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // The first dimension for input is batch.
    const int64 depth_raw = input.dim_size(3);
    OP_REQUIRES(context,
                FastBoundsCheck(depth_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("depth is too large"));
    const int depth = static_cast<int>(depth_raw);


    TensorShape out_shape = input.shape();
        // ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "BoundAngleIndeces: "
            << "input_cols = " << input_cols
            << ", input_rows = " << input_rows;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) return;

    const T *in = input.flat<T>().data();
    T *out = output->flat<T>().data();
    const T *nb = numBins.flat<T>().data();
    const size_t dataLength = (size_t)input.NumElements();

	BoundAngleIndecesFunctor<Device, T>()(context->eigen_device<Device>(), in, out, nb, dataLength);
  }

 private:
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(BoundAngleIndecesOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BoundAngleIndeces").Device(DEVICE_CPU).TypeConstraint<int32>("T"), \
      BoundAngleIndecesOp<CPUDevice, T>);
REGISTER_CPU(int32);
// REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BoundAngleIndeces").Device(DEVICE_GPU).TypeConstraint<int32>("T"), \
      BoundAngleIndecesOp<GPUDevice, T>);
REGISTER_GPU(int32);
// REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

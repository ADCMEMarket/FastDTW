#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include "Dtw_.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("Dtw")

.Input("s : double")
.Input("t : double")
.Input("fast : int32")
.Output("l : double")
.Output("p : int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle s_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &s_shape));
        shape_inference::ShapeHandle t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &t_shape));
        shape_inference::ShapeHandle fast_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &fast_shape));

        c->set_output(0, c->Scalar());
        c->set_output(1, c->Matrix(-1,2));
    return Status::OK();
  });

REGISTER_OP("DtwGrad")

.Input("grad_l : double")
.Input("l : double")
.Input("p : int32")
.Input("s : double")
.Input("t : double")
.Input("fast : int32")
.Output("grad_s : double")
.Output("grad_t : double")
.Output("grad_fast : int32");


class DtwOp : public OpKernel {
private:
  
public:
  explicit DtwOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(3, context->num_inputs());
    
    
    const Tensor& s = context->input(0);
    const Tensor& t = context->input(1);
    const Tensor& fast = context->input(2);
    
    
    const TensorShape& s_shape = s.shape();
    const TensorShape& t_shape = t.shape();
    
    int n1 = s_shape.dim_size(0), n2 = t_shape.dim_size(0);
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    const double* s_tensor = s.flat<double>().data();
    const double* t_tensor = t.flat<double>().data();

    int use_fast = *fast.flat<int32>().data();

    DTW dtw(s_tensor, n1, t_tensor, n2, use_fast);
    dtw.forward(context);   

    // implement your forward function here 
  }
};
REGISTER_KERNEL_BUILDER(Name("Dtw").Device(DEVICE_CPU), DtwOp);



class DtwGradOp : public OpKernel {
private:
  
public:
  explicit DtwGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_l = context->input(0);
    const Tensor& l = context->input(1);
    const Tensor& p = context->input(2);
    const Tensor& s = context->input(3);
    const Tensor& t = context->input(4);
    
    
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& t_shape = t.shape();
    
    
    DCHECK_EQ(grad_l_shape.dims(), 0);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 2);
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_s_shape(s_shape);
    TensorShape grad_t_shape(t_shape);

    int n1 = s_shape.dim_size(0), n2 = t_shape.dim_size(0), np = p_shape.dim_size(0);
            
    // create output tensor
    
    Tensor* grad_s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_s_shape, &grad_s));
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_t_shape, &grad_t));
    
    // get the corresponding Eigen tensors for data access
    
    auto s_tensor = s.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto l_tensor = l.flat<double>().data();
    auto p_tensor = p.flat<int32>().data();
    auto grad_s_tensor = grad_s->flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();   

    // implement your backward function here 
    backward(
      grad_s_tensor, grad_t_tensor, grad_l_tensor, 
      s_tensor, n1, t_tensor, n2, p_tensor, np
    );
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DtwGrad").Device(DEVICE_CPU), DtwGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class DtwOpGPU : public OpKernel {
private:
  
public:
  explicit DtwOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& s = context->input(0);
    const Tensor& t = context->input(1);
    
    
    const TensorShape& s_shape = s.shape();
    const TensorShape& t_shape = t.shape();
    
    
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape l_shape({});
    TensorShape p_shape({-1,2});
            
    // create output tensor
    
    Tensor* l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, l_shape, &l));
    Tensor* p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, p_shape, &p));
    
    // get the corresponding Eigen tensors for data access
    
    auto s_tensor = s.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto l_tensor = l->flat<double>().data();
    auto p_tensor = p->flat<int32>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Dtw").Device(DEVICE_GPU), DtwOpGPU);

class DtwGradOpGPU : public OpKernel {
private:
  
public:
  explicit DtwGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_l = context->input(0);
    const Tensor& grad_p = context->input(1);
    const Tensor& l = context->input(2);
    const Tensor& p = context->input(3);
    const Tensor& s = context->input(4);
    const Tensor& t = context->input(5);
    
    
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& grad_p_shape = grad_p.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& t_shape = t.shape();
    
    
    DCHECK_EQ(grad_l_shape.dims(), 0);
    DCHECK_EQ(grad_p_shape.dims(), 2);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 2);
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_s_shape(s_shape);
    TensorShape grad_t_shape(t_shape);
            
    // create output tensor
    
    Tensor* grad_s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_s_shape, &grad_s));
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_t_shape, &grad_t));
    
    // get the corresponding Eigen tensors for data access
    
    auto s_tensor = s.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto grad_p_tensor = grad_p.flat<int32>().data();
    auto l_tensor = l.flat<double>().data();
    auto p_tensor = p.flat<int32>().data();
    auto grad_s_tensor = grad_s->flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DtwGrad").Device(DEVICE_GPU), DtwGradOpGPU);

#endif
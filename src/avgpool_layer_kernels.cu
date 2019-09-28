#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}

#ifdef Dtype
__device__ void forward_avgpool_layer_Dtype_device(int out_index, int k, int b, int w, int h, int c, Dtype *input, Dtype *output){
    int i;
    Dtype2 accum = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        accum += input[in_index];
    }
    accum /= w*h;
    output[out_index] = (Dtype)accum;
}
#endif

__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, void *in, void *out, int true_q)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int out_index = (k + c*b);
#ifdef Dtype
    if(true_q){
        forward_avgpool_layer_Dtype_device(out_index, k, b, w, h, c, (Dtype *)in, (Dtype *)out);
        return;
    }
#endif
    int i;
    float *input = (float *)in;
    float *output = (float *)out;
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

#ifdef Dtype
    if(net.true_q){
        forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, (void *)net.input_q_gpu, (void *)layer.output_gpu, 1);
    }else
#endif
    {
        forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, (void *)net.input_gpu, (void *)layer.output_gpu, 0);
    }
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(cudaPeekAtLastError());
}


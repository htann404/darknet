#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

#ifdef Dtype
__global__ void yoloswag420blazeit360noscope_Dtype(Dtype *input, int size, float *rand, float prob)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size && rand[id] < prob) input[id] = 0;
}
#endif

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */
#ifdef Dtype
    if (net.true_q){
        yoloswag420blazeit360noscope_Dtype<<<cuda_gridsize(size), BLOCK>>>(net.input_q_gpu, size, layer.rand_gpu, layer.probability);
    }else
#endif
    {
        yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    }
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
}

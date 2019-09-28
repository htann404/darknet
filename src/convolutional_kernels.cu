#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabsf(input[i*size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += fabsf(weights[f*size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    binarize_weights_kernel<<<cuda_gridsize(n), BLOCK>>>(weights, n, size, binary);
    check_error(cudaPeekAtLastError());
}

#ifdef Dtype
__global__ void make_bigger_col_kernel(Dtype *D, int N, Dtype *S,
                                    int ROW, int COL, int DCOL)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int row_idx = i/COL;
    int col_idx = i%COL;
    D[row_idx*DCOL + col_idx] = S[row_idx*COL + col_idx];
}

void make_bigger_col_gpu(Dtype *d, Dtype *s, int row, int col, int drow, int dcol)
{
    int n = row*col;
    //assert(dcol >= col);
    fill_gpu_Dtype(drow*dcol, 0, d, 1);
    make_bigger_col_kernel<<<cuda_gridsize(n), BLOCK>>>(d, n, s, row, col, dcol); 
    check_error(cudaPeekAtLastError());
}

__global__ void make_smaller_col_kernel(Dtype2 *D, int N, Dtype2 *S,
                                     int ROW, int COL, int DCOL)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int row_idx = i/COL;
    int col_idx = i%COL;
    if (col_idx >= DCOL) return;
    D[row_idx*DCOL + col_idx] = S[row_idx*COL + col_idx];
}

void make_smaller_col_gpu(Dtype2 *d, Dtype2 *s, int row, int col, int dcol)
{
    int n = row*col;
    //assert(dcol <= col);
    make_smaller_col_kernel<<<cuda_gridsize(n), BLOCK>>>(d, n, s, row, col, dcol); 
    check_error(cudaPeekAtLastError());
}

void forward_quantized_convolutional_layer_gpu(convolutional_layer *l, network* net)
{
    int i,j;

    fill_gpu_Dtype(l->outputs*l->batch, 0, l->output_q_gpu, 1);

    quantize_params *q = l->quantize;
    if(!q) error("Convolutional layer: quantized params not found!");


    int m = l->n/l->groups;
    int k = l->size*l->size*l->c/l->groups;
    int n = l->out_w*l->out_h;
    int shamt = (q->w_fl + q->in_fl) - q->out_fl;

    // TODO: cublasGemmEx requires lda, ldb to be multiple of 4!!
    // and A_gpu and B_gpu to be 32-bit aligned!!
    size_t factor = sizeof(Dtype2)/sizeof(Dtype);
    size_t size;

    int col_a = (k % 4 == 0) ? k : (k + 4 - (k % 4));
    size = m*col_a*factor;
    Dtype *new_a = (Dtype *)cuda_make_array_Dtype(0, size);

    int col_b = (n % 4 == 0) ? n : (n + 4 - (n % 4));
    size = col_a*col_b*factor;
    Dtype *new_b = (Dtype *)cuda_make_array_Dtype(0, size);

    size = m*col_b*factor;
    Dtype2 *new_c = (Dtype2 *)cuda_make_array_Dtype(0, size);

    if (l->groups != 1)
        fprintf(stderr, "Warning, group convolution is not supported\n");
    for(i = 0; i < l->batch; ++i){
        for(j = 0; j < l->groups; ++j){
            Dtype *a = q->weight_q_gpu + j*l->nweights/l->groups;
            Dtype *b = (Dtype *)net->workspace;
            Dtype2 *c = (Dtype2 *)l->output_q_gpu + (i*l->groups + j)*n*m;
            Dtype *im = net->input_q_gpu + (i*l->groups + j)*l->c/l->groups*l->h*l->w;

            if (l->size == 1) {
                b = im;
            } else {
                im2col_Dtype_gpu(im, l->c/l->groups, l->h, l->w, l->size, l->stride, l->pad, b);
            }
 
            // cublasGemmEx requires lda, ldb to be multiple of 4!
            // and A_gpu and B_gpu to be 32-bit aligned!

            if (col_a == k && col_b == n){
                gemm_gpu_Dtype(0,0,m,n,k,1,a,k,b,n,1,c,n);
            }else{
                if (i==0 || l->groups != 1){
                    make_bigger_col_gpu(new_a, a, m, k, m, col_a);
                }
                make_bigger_col_gpu(new_b, b, k, n, col_a, col_b);

                gemm_gpu_Dtype(0, 0,m,col_b,col_a,1,new_a,col_a,new_b,col_b,1,new_c,col_b);
                make_smaller_col_gpu(c, new_c, m, col_b, n);
            }
/*
#ifdef QDEBUG
    printf("INP: ");
    Dtype2 tmp[1024];
    float div = pow(2, q->w_fl+q->in_fl);
    cuda_pull_array((float*)c, (float*)tmp, 1024/4);
    for (int jj=0; jj<100; ++jj){
        printf("%.2f ", (float)tmp[jj]/div);
    }
    printf("|| %d %d %d, %d %d\n", m, n, k, col_a, col_b);
#endif
*/
        }
    }
    cuda_free_Dtype((Dtype *)new_a);
    cuda_free_Dtype((Dtype *)new_b);
    cuda_free_Dtype((Dtype *)new_c);
/*
#ifdef QDEBUG
    cuda_pull_array((float*)l->output_q_gpu, (float*)l->output_q, l->outputs > 0 ? l->outputs : 100);
    Dtype2* ptr = (Dtype2*)l->output_q;
    FILE *fid=fopen("gemm.out_gpu", "a");
    fprintf(fid, "%d %d %d, %d %d\n", m,n,k,col_a,col_b);
    float div = pow(2, shamt);
    for(int k=0; k<100; ++k){
        float val = (float)ptr[k];
        fprintf(fid, "%f ", val/div);
    }
    fprintf(fid, "\n");
    fclose(fid);
#endif
*/
    if (l->batch_normalize) {
        forward_batchnorm_layer_gpu(*l, *net);
    }
 
    align_Dtype2_radix_gpu((Dtype2 *)l->output_q_gpu, l->batch*n*m, shamt);
    add_bias_Dtype2_gpu((Dtype2 *)l->output_q_gpu, q->bias_q_gpu, l->batch, l->n, n);
    shrink_Dtype2_to_Dtype_gpu((Dtype2 *)l->output_q_gpu, l->batch*n*m, 0);
/*
#ifdef QDEBUG
    printf("OUT: ");
    Dtype tmp[1024];
    float div = pow(2, q->out_fl);
    cuda_pull_array((float*)l->output_q_gpu, (float*)tmp, 1024/4);
    for (int jj=0; jj<100; ++jj){
        printf("%.1f ", (float)tmp[jj]/div);
    }
    printf("\n");
#endif
*/
    activate_array_Dtype_gpu(l->output_q_gpu, l->outputs*l->batch, l->activation);

}
#endif

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
#ifdef Dtype
    if (net.true_q){
        forward_quantized_convolutional_layer_gpu(&l, &net);
        return;
    }
#endif
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

    quantize_params *q = l.quantize;
    if(q) {
        quantize_gpu(net.input_gpu, l.inputs*l.batch, q->in_bw, q->in_fl, q->mode, q->a_type);
    }
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }

            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
/*
#ifdef QDEBUG
            printf("INP: ");
            float tmp[1024];
            cuda_pull_array(c, tmp, 1024/4);
            for (int jj=0; jj<100; ++jj){
                printf("%.2f ", tmp[jj]);
            }
            printf("\n");
#endif
*/
        }
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
/*
#ifdef QDEBUG
    printf("OUT: ");
    float tmp[1024];
    cuda_pull_array(l.output_gpu, tmp, 1024/4);
    for (int jj=0; jj<100; ++jj){
        printf("%.2f ", tmp[jj]);
    }
    printf("\n");
#endif
*/
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    if(q) {
        quantize_gpu(l.output_gpu, l.outputs*l.batch, q->out_bw, q->out_fl, q->mode, q->a_type);
    }
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);

#ifdef Dtype
    float *space=0;
    //float *weights=0;
    if (net.true_q){
        int size = l.batch*l.c*l.h*l.w;
        space = cuda_make_array(0, size);
        //weights = cuda_make_array(0, l.nweights);
        copy_Dtype_to_float_gpu(space, (void *)net.input_q_gpu, size,
                                l.quantize->in_fl, sizeof(Dtype));
        //copy_Dtype_to_float_gpu(weights, (void *)l.quantize->weight_q_gpu, l.nweights,
        //                        l.quantize->w_fl, sizeof(Dtype));
        //weights = l.weights_gpu;
        gradient_array_Dtype_gpu(l.output_q_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    }else
#endif
    {
        gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    }

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im;
#ifdef Dtype
            if(net.true_q){ 
                im  = space + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            }else
#endif
                im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
/*
#ifdef Dtype
                if(net.true_q)
                    a = weights + j*l.nweights/l.groups;
                else
#endif
*/
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }

#ifdef Dtype
    if(space)   cuda_free(space);
    //if(weights) cuda_free(weights);
#endif
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        if(l.weight_prune_mask_gpu)
            mul_gpu(l.nweights, l.weight_prune_mask_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);
        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}



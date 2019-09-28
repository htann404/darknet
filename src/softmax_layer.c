#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        float *input = net.input;
#ifdef Dtype
        quantize_params *q = l.quantize;
        if (net.true_q && net.train){
            copy_Dtype_to_float_cpu(net.workspace, net.input_q, l.batch*l.c*l.w*l.h, q->out_fl, sizeof(Dtype));
            input = net.workspace; 
        }
        /*
        for (int jj=0; jj<10; ++jj){
            printf("%f ", input[jj]);
        }
        printf("\n");*/
        if (net.true_q && net.train==0){
            if(l.spatial){
                softmax_cpu_Dtype(net.input_q, l.c, l.batch, l.inputs, l.w*l.h, 1,
                                                l.w*l.h, l.temperature, l.output);
            }else{
                softmax_cpu_Dtype(net.input_q, l.inputs/l.groups, l.batch, l.inputs,
                                    l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
            }
        }else
#endif
        {
            if(l.spatial){
                softmax_cpu(input, l.c, l.batch, l.inputs, l.w*l.h, 1,
                            l.w*l.h, l.temperature, l.output);
            }else{
                softmax_cpu(input, l.inputs/l.groups, l.batch, l.inputs,
                            l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
            }
        }
    }
    
    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
        /*
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
            count += group_size;
        }
        */
    } else {
        float *input = net.input_gpu;
#ifdef Dtype
        quantize_params *q = l.quantize;
        if (net.true_q){
            copy_Dtype_to_float_gpu(net.workspace, net.input_q_gpu, l.batch*l.c*l.w*l.h, q->out_fl, sizeof(Dtype));
            input = net.workspace;
        }
#endif
        if(l.spatial){
            softmax_gpu(input, l.c, l.batch*l.c,
                        l.inputs, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
            softmax_gpu(input, l.inputs/l.groups, l.batch,
                        l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        if(l.softmax_tree){
            mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
            mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif

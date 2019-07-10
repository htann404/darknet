#ifndef COL2IM_H
#define COL2IM_H
#include "darknet.h"

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef Dtype
void col2im_cpu_Dtype(Dtype2* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, Dtype2* data_im);
#endif

#ifdef GPU
void col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);
#endif
#endif

#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <float.h>

typedef enum{
    MAGNITUDE, PERCENTAGE, HALFWAY
} PRUNE_TYPE;

float THRESH=0.04;
float MAX_ACCURACY_DIFF=.002;

// Implement iterative pruning:
void prune(network *net, PRUNE_TYPE type, float percent, int first_layer, int last_layer){
    int n = net->n;
    int i, j, num_weights;
    float w;
    float *weights, *weights_gpu;
    
    if (type==MAGNITUDE){
        fprintf(stderr, "Performing magnitude-based pruning.\n");
    }else if(type==PERCENTAGE){
        fprintf(stderr, "Performing percentage-based pruning.\n");
    }else{
        fprintf(stderr, "Performing halfway pruning.\n");
    }
    // don't prune the first layer:
    int first = 0;
    while(first < n){
        if (net->layers[first].weights) {
            break;
        }
        ++first;
    }
    for (i=0; i<n; ++i){
        if (i==first && !first_layer) continue;    
        //fprintf(stderr, "Pruning layer %d\n", i);    
        layer *l = &(net->layers[i]);
        if(!l->weights)
            continue;
        // true for conv, deconv and fc layers.
        num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
        assert(num_weights > 0);
        quantize_params *q = l->quantize;
        weights = (q)? ((q->weight_copy)? q->weight_copy : l->weights) : l->weights;
        
        // if we don't prune the last layer
        j = i;
        while(++j < n){
            if (net->layers[j].weights) {
                break;
            }
        }
        if (j==n && !last_layer) continue;
        
        
#ifdef GPU
        if(gpu_index >= 0){
            weights_gpu = (q)? ((q->weight_copy_gpu)? q->weight_copy_gpu 
                                : l->weights_gpu) : l->weights_gpu;
            cuda_pull_array(weights_gpu, weights, num_weights);
            cuda_pull_array(l->weight_prune_mask_gpu, l->weight_prune_mask, num_weights);
        }
#endif
        if (type == MAGNITUDE) {
            for (j=0; j<num_weights; ++j){
                w = fabs(weights[j]);
                if (w <= THRESH){
                    weights[j] = 0;
                    l->weight_prune_mask[j] = 0;
                }
            }
        } else if (type==PERCENTAGE) {
            float *w_copy = malloc(num_weights*sizeof(float));
            assert(percent <= 1);
            assert(percent >= 0);
            int k = num_weights - num_weights*(1-percent);

            memcpy(w_copy, weights, num_weights*sizeof(float));
            for (j=0; j<num_weights; ++j){
                if(w_copy[j]==0){
                    w_copy[j] = FLT_MAX;
                }
            }
            selection(w_copy, num_weights, k, 0);
            float kth_weight = w_copy[k];
            for (j=0; j<num_weights; ++j){
                w = fabs(weights[j]);
                if (w <= kth_weight){
                    weights[j] = 0;
                    l->weight_prune_mask[j] = 0;
                }
            }
            free(w_copy);
        } else if (type==HALFWAY) {
            float *w_copy = malloc(num_weights*sizeof(float));
            memcpy(w_copy, weights, num_weights*sizeof(float));
/*          for (j=0; j<num_weights; ++j)
                if(w_copy[j])
                    w_copy[j] = fabs(w_copy[j]);
                
            float maxval=FLT_MIN, minval=FLT_MAX;
            for(j = 0; j<num_weights; ++j){
                if(w_copy[j]){
                    if(w_copy[j] > maxval)
                        maxval = w_copy[j];
                    if(w_copy[j] < minval)
                        minval = w_copy[j];
                }
            }*/
            // usually long-tail dist, so we do not take absolute max
            int k = 0.05*num_weights;
            selection(w_copy, num_weights, k, 1);

            float halfway = w_copy[k]/2;
            for (j=0; j<num_weights; ++j){
                w = fabs(weights[j]);
                if (w < halfway){
                    weights[j] = 0;
                    l->weight_prune_mask[j] = 0;
                }
            }
            free(w_copy);
        }
#ifdef GPU
        if(gpu_index >= 0){
            cuda_push_array(weights_gpu, weights, num_weights);
            cuda_push_array(l->weight_prune_mask_gpu, l->weight_prune_mask, num_weights);
        }
#endif    
    }
}

// Initialize the prune mask:
void init_prune_mask(network *net, int from_scratch){
    int i, j, num_weights;
    for (i=0; i<net->n; ++i) {
        layer *l = &(net->layers[i]);
        if (!l->weights)
            continue;
        // true for conv, deconv and fc layers.
        num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
        assert(num_weights > 0);
        if (!l->weight_prune_mask)
            l->weight_prune_mask = malloc(num_weights*sizeof(float));
        for (j=0; j<num_weights; ++j){
            l->weight_prune_mask[j] = (from_scratch) ? 1. : (float)(l->weights[j]!=0);
        }
#ifdef GPU
        if (gpu_index >= 0){
            if(!l->weight_prune_mask_gpu)
                l->weight_prune_mask_gpu = cuda_make_array(l->weight_prune_mask, num_weights);
            else
                cuda_push_array(l->weight_prune_mask_gpu, l->weight_prune_mask, num_weights);
        }
#endif
    }

}
// TODO: Check the case when l->groups > 1
void compress_weights(network *net)
{
    fprintf(stderr, "Compressing network weights\n");
    int i, j, k, rows, cols;
    int count, nnz, num_weights;
    for (i=0; i<net->n; ++i) {
        layer *l = &(net->layers[i]);
        // test convolutional for now:
        if(!l->weights)
            continue;

        num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
        if (l->type == CONVOLUTIONAL) {
            rows = l->n/l->groups;
            cols = l->size*l->size*l->c/l->groups;
        }else if(l->type == DECONVOLUTIONAL){
            rows = l->c;
            cols = l->size*l->size*l->n; 
        }else if(l->type == CONNECTED){
            rows = l->outputs;
            cols = l->inputs;
        }else{
            continue;
        }

        //fprintf(stderr, "num_weights %d, dimensions: %dx%d\n", num_weights, rows, cols);
        assert(num_weights == rows*cols);

        nnz = 0;
        for (j=0; j<num_weights; ++j)
            if (l->weights[j]!=0)
                ++nnz;

        if (!l->weights_c)
            l->weights_c = calloc(1, sizeof(compressed_weights));
        compressed_weights *w_c = l->weights_c;
        w_c->nnz = nnz;

#ifdef Dtype
        if (net->true_q){
            if (!l->quantize || !l->quantize->weight_q)
                error("Error in compress weights. true_q is set, but quantized weights not found!");
            if (!w_c->w_q)
                w_c->w_q = malloc(nnz*sizeof(Dtype));
        }else
#endif
        {
            if (!w_c->w)
                w_c->w = malloc(nnz*sizeof(float));
        }

        if (!w_c->jw)
            w_c->jw = malloc(nnz*sizeof(int));    
        
        if (l->type == CONVOLUTIONAL || l->type == CONNECTED) {
            w_c->type = CSR;
            w_c->n = rows;
            if (!w_c->iw)
                w_c->iw = malloc((w_c->n+1)*sizeof(int));

            count = 0;
            w_c->iw[0] = 0;
            for (j=0; j<rows; ++j){
                for (k=0; k<cols; ++k){
                    if (l->weights[j*cols + k] != 0){
#ifdef Dtype
                        if (!net->true_q)
                            w_c->w[count] = l->weights[j*cols + k];
                        else
                            w_c->w_q[count] = l->quantize->weight_q[j*cols + k];
#else
                        w_c->w[count] = l->weights[j*cols + k];
#endif
                        w_c->jw[count] = k;
                        ++count;
                    }
                }
                w_c->iw[j+1] = count;
            }
        }else if(l->type == DECONVOLUTIONAL){
            w_c->type = CSC;
            w_c->n = cols;
            if (!w_c->iw)
                w_c->iw = malloc((w_c->n+1)*sizeof(int));

            count = 0;
            w_c->iw[0] = 0;
            for (j=0; j<cols; ++j){
                for (k=0; k<rows; ++k){
                    if (l->weights[k*cols + j] != 0){
#ifdef Dtype
                        if (!net->true_q)
                            w_c->w[count] = l->weights[k*cols + j];
                        else
                            w_c->w_q[count] = l->quantize->weight_q[k*cols + j];
#else
                        w_c->w[count] = l->weights[k*cols + j];
#endif
                        w_c->jw[count] = k;
                        ++count;
                    }
                }
                w_c->iw[j+1] = count;
            }
        }
/*
        for(j = 0; j < rows; ++j){
        int start = w_c->iw[j];
        int end = w_c->iw[j+1];
        fprintf(stderr, "j %d s %d e %d\n", j, start, end);
        for(k = start; k < end; ++k)
            fprintf(stderr, "%.1f ",w_c->w[k]);
        fprintf(stderr, "\n");
        for(k = start; k < end; ++k)
            fprintf(stderr, "%d ",w_c->jw[k]);
        fprintf(stderr, "\n");
        }
*/
        fprintf(stderr, "layer: %d, type: %d, num weights %d, nnz: %d\n", i, l->type, num_weights, nnz);
        assert(count == nnz);
    }    
}


void count_zeros(network *net){
    int i,j,count,num_weights;
    int total_w=0, total_acts=0;
    int total_wz=0, total_actsz=0;
    for (i=0; i<net->n; ++i) {
        layer *l = &(net->layers[i]);
        if(!l->weights)
            continue;
        num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
        total_w += num_weights;

        count = 0;
        for (j=0; j<num_weights; ++j)
            if (l->weights[j]==0)
                ++count;
        total_wz += count;
        fprintf(stderr, "Layer %d, Type %d, Num Weights: %d, Fraction Zeros: %.4f\n",
                i, l->type, num_weights, ((float)count)/num_weights);
        
        if(!l->outputs)
            continue;

        total_acts += l->outputs;
        count = 0;
#ifdef GPU
        if(gpu_index >= 0)
            cuda_pull_array(l->output_gpu, l->output, l->outputs);
#endif
        for (j=0; j<l->outputs; ++j) {
#ifdef Dtype
            if (net->true_q) {
                if (l->output_q[j]==0)
                    ++count;
            }else
#endif
            {
                if (l->output[j]==0)
                    ++count;
            }
        }
        
        total_actsz += count;
        fprintf(stderr, "\t\tNum Activations: %d, Fraction Zeros: %.4f\n",
                            l->outputs, ((float)count)/l->outputs );
    }
    fprintf(stderr, "Total Num Weights: %d, Fraction Zeros: %.4f\n",
            total_w, ((float)total_wz)/total_w);
    fprintf(stderr, "Total Num Activations: %d, Fraction Zeros: %.4f\n",
                total_acts, ((float)total_actsz)/total_acts );
}


void prune_segmenter(network *net, char *backup_directory, char *base, load_args args)
{
    float avg_loss = -1;
    
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    int N = args.m;
    printf("%d\n", N);
    
    float orig_acc[NUM_SEG_ACCURACY_ELEMENTS];
    float cur_acc[NUM_SEG_ACCURACY_ELEMENTS];
    run_and_calc_seg_accuracy(net, &args, N, orig_acc);
    
    init_prune_mask(net,1);
    prune(net, MAGNITUDE, 0, 1, 1);

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
        if(!net->quantized){
            loss = train_network(net, train);
        }else{
            loss = train_network_quantized(net, train); 
        }

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", 
                get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, 
                get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.pruned_weights",backup_directory,base, epoch);
            save_weights(net, buff);
            // SEGMENTATION: Using E1 error rate as the target for now:
            run_and_calc_seg_accuracy(net, &args, N, cur_acc);
            if (cur_acc[3] - orig_acc[3] < MAX_ACCURACY_DIFF
                && cur_acc[1] - orig_acc[1] > -MAX_ACCURACY_DIFF
                && cur_acc[0] - orig_acc[0] > -MAX_ACCURACY_DIFF)
                prune(net, PERCENTAGE, 0.02, 0, 0);
                count_zeros(net);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.pruned_backup",backup_directory,base);
            save_weights(net, buff);
        }
    }

    char buff[256];
    sprintf(buff, "%s/%s.pruned_weights", backup_directory, base);
    save_weights(net, buff);

    count_zeros(net);
    run_and_calc_seg_accuracy(net, &args, N, cur_acc);
    
    fprintf(stderr, "Accuracy Comparison:\tR\tP\tF1\tE1\tE2\n");
    fprintf(stderr, "Original Model:\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", orig_acc[0],
                    orig_acc[1], orig_acc[2], orig_acc[3], orig_acc[4]);
    fprintf(stderr, "Pruned Model\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", cur_acc[0],
                    cur_acc[1], cur_acc[2], cur_acc[3], cur_acc[4]);
}


// almost verbatim as train_classifier from classifier.c
void prune_classifier(network *net, char *backup_directory, char *base, load_args args)
{
    float avg_loss = -1;
    double time;
    int N = args.m;

    printf("%d\n", N);
    printf("%d %d\n", args.min, args.max);

    float orig_acc[2] = {0.8911, 0.9616};//{0.84272, 0.93614}->model with leaky activ;
    float cur_acc[2];
    //validate_classifier_from_net(*net, 2, args, orig_acc);
    init_prune_mask(net,1);
    prune(net, MAGNITUDE, 0, 1, 1);
    count_zeros(net);
    prune(net, HALFWAY, 0, 0, 0);
    count_zeros(net);
    validate_classifier_from_net(*net, 2, args, cur_acc);

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
            printf("Resizing\n");
            int dim = (rand() % 11 + 4) * 32;
            
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
            printf("%d %d\n", args.min, args.max);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            resize_network(net, dim, dim);
        }
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
        if (!net->quantized) {
            loss = train_network(net, train);
        }else{
            loss = train_network_quantized(net, train); 
        }

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", 
                get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, 
                get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.pruned_weights",backup_directory,base, epoch);
            save_weights(net, buff);
            
            // Using top-1 for pruning
            validate_classifier_from_net(*net, 2, args, cur_acc);
            if(orig_acc[0] - cur_acc[0] < MAX_ACCURACY_DIFF){
                prune(net, PERCENTAGE, 0.02, 0, 0);
                count_zeros(net);
            }
        } 
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.pruned_backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.pruned_weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);
    
    count_zeros(net);
    validate_classifier_from_net(*net, 2, args, cur_acc);
    fprintf(stderr, "Original Model:\t%.4f\t%.4f\n", orig_acc[0],    orig_acc[1]);
    fprintf(stderr, "Pruned Model\t%.4f\t%.4f\n", cur_acc[0], cur_acc[1]);        
}


// TODO: make more generic and more data types than just SEGMENTATION_DATA
// TODO: support more than conv, deconv, and fc
void iterative_pruning(TASK_TYPE task, char *datacfg, char *cfgfile, char *weightfile, 
                        char *quantized_cfg, int gpu, int clear)
{
    int i;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);

    srand(time(0));
    int seed = rand();
    srand(seed);
#ifdef GPU
    if(gpu >= 0)
        cuda_set_device(gpu);
#endif
    network *net = load_network(cfgfile, weightfile, clear);
    srand(time(0));

    // Initialize the quantization parameters if given:
    if (quantized_cfg) {
        allocate_quantized_weight_copy(net);
        read_quantized_net_cfg(net, quantized_cfg);
        net->quantized = 1;
        fprintf(stderr, "Done reading quantization config file.\n");
    }

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

    list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);

    if (tree) net->hierarchy = read_tree(tree);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    load_args args = set_load_args(task, net, paths, options, N);

    if (task == SEGMENTATION){
        prune_segmenter(net, backup_directory, base, args);
    }else{
        prune_classifier(net, backup_directory, base, args);
    }

    if(args.labels) free_ptrs((void**)(args.labels), args.classes);
    if(paths) free_ptrs((void**)paths, plist->size);
    if(plist) free_list(plist);
    for (i=0; i<net->n; ++i) {
        if(net->layers[i].weight_prune_mask)
            free(net->layers[i].weight_prune_mask);
#ifdef GPU
        if(net->layers[i].weight_prune_mask_gpu)
            cuda_free(net->layers[i].weight_prune_mask_gpu);    
#endif
    }
    if(net) free_network(net);
    if(base) free(base);
}

void write_compressed_weights(network *net_in, char *backup_directory, char *base,
                    char *datacfg, char *cfgfile, char *weightfile, char *quantized_cfg)
{
    network *net = net_in;
    if (!net){
        net = load_network(cfgfile, weightfile, 0);
        
        if (quantized_cfg){
            read_quantized_net_cfg(net, quantized_cfg);
            fprintf(stderr, "Done reading quantization config file.\n");
        }
    }
    if (!base) base = basecfg(cfgfile);
    if (!backup_directory){
        list *options = read_data_cfg(datacfg);
        backup_directory = option_find_str(options, "backup", "/backup/");
    }

    char buff[256];
    sprintf(buff, "%s/%s.compressed_weights", backup_directory, base);

    fold_batch_norm_params(net);
    compress_weights(net);    
    save_compressed_weights(net, buff, 0, net->n);
}

void count_using_model_cfg(int q, char *datacfg, char *cfgfile, char *weightfile, 
                           char* quantized_cfg, char *filename, int compress)
{
    network *net = load_network(cfgfile, weightfile, 0);
    net->train = 0;
    if (net->batch > 1){
        set_batch_network(net, 1);
        fprintf(stderr, "Batch size set to 1.\n");
    }
    //fold_batch_norm_params(net);
    // Initialize the quantization parameters:
    if (quantized_cfg){
#ifdef Dtype
        net->true_q = q;
#endif
        read_quantized_net_cfg(net, quantized_cfg);
        fprintf(stderr, "Done reading quantization config file.\n");
    }

    if (compress)
        compress_weights(net);

    if (filename){
        float *predictions;
        image im, sized;
        double time;
#ifdef Dtype
        image_Dtype imD, sizedD;
        if (q) {
            imD = load_image_color_Dtype(filename, 0, 0);
            sizedD = center_crop_image_Dtype(imD, net->w, net->h);
            time = clock();
            predictions = network_predict_Dtype(net, sizedD.data);
        } else
#endif
        {
            im = load_image_color(filename, 0, 0);
            sized = center_crop_image(im, net->w, net->h);
            time = clock();
            predictions = network_predict(net, sized.data);
        }
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        printf("Predicted: %f\n", predictions[0]);
		for (int k=0; k<10; ++k)
			printf("%f ", predictions[k]);
		printf("\n");
#ifdef Dtype
        if(net->true_q){
            free_image_Dtype(imD);
            free_image_Dtype(sizedD);
        }else
#endif
        {
            free_image(im);
            free_image(sized);
        }
    }
    count_zeros(net);
    free_network(net);
}

void run_pruning(int argc, char **argv)
{
    if(argc < 6){
        fprintf(stderr, "usage: %s %s function [options] data_cfg net_cfg weights\n", argv[0], argv[1]);
        fprintf(stderr, "\t\t\t[quantization_cfg] [test_input]\n");
        return;
    }

    int gpu = gpu_index;

    int clear = find_arg(argc, argv, "-clear");
    int nq = find_arg(argc, argv, "-no-quantize-cfg");
    int true_q = find_arg(argc, argv, "-true-quantize");
    int compress = find_arg(argc, argv, "-compress");
    
    // default to classification
    char *ttype = NULL;
    TASK_TYPE task=CLASSIFICATION;
    if(find_char_arg(argc, argv, "-task", ttype)){
        if (strcmp(ttype, "segmentation")==0){
            task = SEGMENTATION;
        }else{
            // other kinds of task
        }
    }

    float thresh = find_float_arg(argc, argv, "-thresh", 0.); 
    if(thresh != 0){
        THRESH = thresh;
    }

    float maxdiff = find_float_arg(argc, argv, "-maxdiff", 0.); 
    if(maxdiff != 0){
        MAX_ACCURACY_DIFF = maxdiff;
    }

    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *quantized_cfg = (argc > 6) ? argv[6] : 0; 
    char *filename = (argc > 7) ? argv[7]: 0;
    if (nq){
        filename = quantized_cfg;
        quantized_cfg = NULL;
        true_q = 0;
    }

    if(0==strcmp(argv[2], "count"))
        count_using_model_cfg(true_q, data, cfg, weights, quantized_cfg, filename, compress);
    else if(0==strcmp(argv[2], "iterative"))
        iterative_pruning(task, data, cfg, weights, quantized_cfg, gpu, clear);
    else if(0==strcmp(argv[2], "compress_weights")){
        if (gpu_index >= 0)
            error("Use -nogpu option to write compressed weights.");
        write_compressed_weights(NULL, NULL, NULL, data, cfg, weights, quantized_cfg);
    }
}



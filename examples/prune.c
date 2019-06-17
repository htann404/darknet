#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#define THRESH 0.04
#define MAX_ACCURACY_DIFF 0.002
typedef enum{
    MAGNITUDE, PERCENTAGE
} PRUNE_TYPE;

// Implement iterative pruning:
void prune(network *net, PRUNE_TYPE type, float percent, int first_layer, int last_layer){
	int n = net->n;
	int i, j, num_weights;
	float w;
	float *weights;
	
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
		float *weights_gpu;
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
				if(w_copy[j]==0)
					w_copy[j] = FLT_MAX;
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
void init_prune_mask(network *net){
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
			l->weight_prune_mask[j] = 1;
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
		
		if (!w_c->w)
			w_c->w = malloc(nnz*sizeof(float));
		if (!w_c->jw)
			w_c->jw = malloc(nnz*sizeof(int));	
		w_c->nnz = nnz;
		if (l->type == CONVOLUTIONAL) {
			w_c->type = CSR;
			w_c->n = rows;
			if (!w_c->iw)
				w_c->iw = malloc((w_c->n+1)*sizeof(int));
			count = 0;
			w_c->iw[0] = 0;
			for (j=0; j<rows; ++j){
				for (k=0; k<cols; ++k){
					if (l->weights[j*cols + k] != 0){
						w_c->w[count] = l->weights[j*cols + k];
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
						w_c->w[count] = l->weights[k*cols + j];
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
	for (i=0; i<net->n; ++i) {
		layer *l = &(net->layers[i]);
		if(!l->weights)
			continue;
		num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
		count = 0;
		for (j=0; j<num_weights; ++j)
			if (l->weights[j]==0)
				++count;
		fprintf(stderr, "Layer %d, Type %d, Num Weights: %d, Fraction Zeros: %.4f\n",
				i, l->type, num_weights, ((float)count)/num_weights);
		
		if(!l->outputs)
			continue;
		count = 0;
#ifdef GPU
		if(gpu_index >= 0)
			cuda_pull_array(l->output_gpu, l->output, l->outputs);
#endif
		for (j=0; j<l->outputs; ++j)
			if (l->output[j]==0)
				++count;
		fprintf(stderr, "\t\tNum Activations: %d, Fraction Zeros: %.4f\n",
							l->outputs, ((float)count)/l->outputs );
	}
}

// this is mostly similar to segmenter_train
// TODO: make more generic and more data types than just SEGMENTATION_DATA
// TODO: support more than conv, deconv, and fc
void iterative_pruning(char *datacfg, char *cfgfile, char *weightfile, char *quantized_cfg, int *gpus, int ngpus, int clear)
{
    int i;
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

	// Initialize the quantization parameters if given:
	if (quantized_cfg) {
		allocate_quantized_weight_copy(net);
		read_quantized_net_cfg(net, quantized_cfg);
		fprintf(stderr, "Done reading quantization config file.\n");
	}

    image pred = get_network_image(net);

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int grayscale = option_find_int(options, "grayscale", 0); 

    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("Number of images: %d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.scale = div;

    args.min = net->min_crop;
    args.max = net->max_crop;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;
    args.classes = classes;
    args.grayscale = grayscale;

    args.paths = paths;
    args.n = net->batch * net->subdivisions * ngpus;
    args.m = N;
    args.type = SEGMENTATION_DATA;

	float orig_acc[NUM_SEG_ACCURACY_ELEMENTS] = {0};
	float cur_acc[NUM_SEG_ACCURACY_ELEMENTS] = {0};
	run_and_calc_seg_accuracy(net, &args, N, orig_acc);

	init_prune_mask(net);
	//prune(net, PERCENTAGE, 0.05, 1, 1);
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
		if(!quantized_cfg){
#ifdef GPU
        	if(ngpus == 1){
            	loss = train_network(net, train);
        	} else {
            	loss = train_networks(nets, ngpus, train, 4);
        	}
#else
        	loss = train_network(net, train);
#endif
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
			run_and_calc_seg_accuracy(net, &args, N, cur_acc);
			// Using E1 error rate as the target for now:
			if (cur_acc[3] - orig_acc[3] < MAX_ACCURACY_DIFF
				&& cur_acc[1] - orig_acc[1] > -MAX_ACCURACY_DIFF
				&& cur_acc[0] - orig_acc[0] > -MAX_ACCURACY_DIFF)
				prune(net, PERCENTAGE, 0.02, 0, 0);

            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
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
	for (i=0; i<net->n; ++i) {
		if(net->layers[i].weight_prune_mask)
			free(net->layers[i].weight_prune_mask);
#ifdef GPU
		if(net->layers[i].weight_prune_mask_gpu)
			cuda_free(net->layers[i].weight_prune_mask_gpu);	
#endif
	}
    if(net) free_network(net);
    if(paths) free_ptrs((void**)paths, plist->size);
    if(plist) free_list(plist);
    if(base) free(base);
}

void count_using_model_cfg(char *datacfg, char *cfgfile, char *weightfile, char* quantized_cfg, char *filename, int compress)
{
	network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
	
	if (compress)
		compress_weights(net);

	// Initialize the quantization parameters:
	if (quantized_cfg){
		read_quantized_net_cfg(net, quantized_cfg);
		fprintf(stderr, "Done reading quantization config file.\n");
	}

	if (filename){
		image im = load_image_color(filename, 0, 0);
        image sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data;
        double time = clock();
        float *predictions = network_predict(net, X);
        printf("Predicted: %f\n", predictions[0]);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        free_image(im);
        free_image(sized);
	}
	count_zeros(net);
	free_network(net);
}

void run_pruning(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [single/iterative] [cfg] [weights]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
	int nq = find_arg(argc, argv, "-no-quantize");
	int compress = find_arg(argc, argv, "-compress");

    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
	char *quantized_cfg = (argc > 6) ? argv[6] : 0; 
    char *filename = (argc > 7) ? argv[7]: 0;
	if (nq){
		filename = quantized_cfg;
		quantized_cfg = NULL;
	}

    if(0==strcmp(argv[2], "count"))
		count_using_model_cfg(data, cfg, weights, quantized_cfg, filename, compress);
    else if(0==strcmp(argv[2], "iterative"))
		iterative_pruning(data, cfg, weights, quantized_cfg, gpus, ngpus, clear);
}



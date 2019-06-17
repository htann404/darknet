#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <math.h>

#define BITWIDTH 8
#ifndef INT_MIN
#define INT_MIN -2147483648
#endif

// Iteratively quantize..
void analyze_activations(network *net, data d, profiler *prof) {
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i,j,k,count;
	float *weight_copy, *bias_copy, *act_copy;
	int num_weights, num_acts;
	*net->seen += batch;
    for(i = 0; i < n; ++i){
		count = 0;
    	for(j = 0; j < net->n; ++j){
        	net->index = j;
        	layer l = net->layers[j];
			l.forward(l, *net);
	    	net->input = l.output;
			if (!l.weights)
				continue;

			fprintf(stderr,"Layer %d, Type: %d\n", j, l.type);		 
			// analyze the weights:
			if (prof[count].max_weight == 0){
				prof[count].layer_index = j;
				prof[count].type = l.type;
				prof[count].batch = batch;
			    prof[count].n = l.out_c;
			    prof[count].c = l.c;
			    prof[count].h = l.out_h;
				prof[count].w = l.out_w;

				num_weights = (l.nweights > 0) ? l.nweights : l.inputs*l.outputs;
				weight_copy = malloc(num_weights*sizeof(float));
				memcpy(weight_copy, l.weights, num_weights*sizeof(float));
				// get the 99% largest:
				k = num_weights - num_weights*0.99;
				selection(weight_copy, num_weights, k, 1);
				prof[count].max_weight = weight_copy[0];
				prof[count].val99_weight = weight_copy[k];				
				fprintf(stderr, "num_weights: %d, k: %d, max: %f, max99: %f\n",
						num_weights, k, prof[count].max_weight, prof[count].val99_weight);
				free(weight_copy);

				// bias:
				num_weights = (l.nbiases) ? l.nbiases : l.outputs;
				bias_copy = malloc(num_weights*sizeof(float));
				memcpy(bias_copy, l.biases, num_weights*sizeof(float));
				k = num_weights - num_weights*0.99;
				selection(bias_copy, num_weights, k, 1);
				prof[count].max_bias = bias_copy[0];
				prof[count].val99_bias = bias_copy[k];
				fprintf(stderr, "num_bias: %d, k: %d, max: %f, max99: %f\n",
						num_weights, k, prof[count].max_bias, prof[count].val99_bias);
				free(bias_copy);
			}
			if (l.output){
				num_acts = l.batch*l.outputs;
				k = num_acts - (num_acts - 1)*0.99;
				act_copy = malloc(num_acts*sizeof(float));
				memcpy(act_copy, l.output, num_acts*sizeof(float));
				selection(act_copy, num_acts, k, 1);
				if (prof[count].max_activation < act_copy[0])
					prof[count].max_activation = act_copy[0];
				if (prof[count].val99_activation < act_copy[k])
					prof[count].val99_activation = act_copy[k];
				fprintf(stderr, "num_activations: %d, k: %d, max: %f, max99: %f\n",
						num_acts, k, prof[count].max_activation, prof[count].val99_activation);
				free(act_copy);
				count++;
			}
    	}
    }
	printf("Done with analysis.\n");
}
// TODO: take care of the case when Batch Norm is declared as a layer
void fold_batch_norm_params(network *net){
	int i,j,k;
	for (i=0; i<net->n; ++i){
		layer *l = &(net->layers[i]);
		if (!l->weights || !l->batch_normalize)
			continue;
		int c = l->out_c;
		int nweights = l->nweights;
		float *mean = l->rolling_mean;
		float *var = l->rolling_variance;
		float *scales = l->scales;
		float *w = l->weights;
		float *biases = l->biases;

		for (j=0; j<c; ++j){
			float denom = sqrt(var[j])+0.000001f;
			float alpha = scales[j]/denom;

			biases[j] -= mean[j]*alpha;
			int start = j*nweights/c;
			int end = (j+1)*nweights/c;
			for (k=start; k<end; ++k){
				w[k] *= alpha;
			}
		}
		l->batch_normalize = 0;
#ifdef GPU
		if (gpu_index >=0){
			cuda_push_array(l->weights_gpu, w, nweights);
			cuda_push_array(l->biases_gpu, biases, c);
		}
#endif
	}
	fprintf(stderr, "Done folding all batchnorm params.\n");
}


// TODO: replace with actually generating a new cfg with quantization params
void write_quantized_net_cfg(network *net, profiler *prof, char *filename) {
	FILE *fid = fopen(filename, "w");
	if(!fid) error("Unable to open file to write quantization cfg.");
	
	int i,count=0;
	ROUNDING_MODE mode = ROUND_NEAREST;
	QUANTIZATION_TYPE a_type = DFP;
	QUANTIZATION_TYPE w_type = DFP;
	int in_bw, in_fl, out_bw, out_fl=INT_MIN, w_bw, w_fl;
	for (i=0; i<net->n; ++i){
		if (!net->layers[i].weights)
			continue;
/*
		fprintf(fid, "%d %d %d %d %d %d %d\n",
					prof[count].layer_index,prof[count].type,
					prof[count].batch, prof[count].n,prof[count].c,
					prof[count].h, prof[count].w);
*/
		in_bw = BITWIDTH;
		in_fl = (out_fl > INT_MIN) ? out_fl : BITWIDTH - 1;
		out_bw = BITWIDTH;
		// we choose fraction length such that the 99 percentile
		// largest value does not saturate:
		out_fl = BITWIDTH - (int)(ceil(log2(prof[count].max_activation)));
		w_bw = BITWIDTH;
		w_fl = BITWIDTH - (int)(ceil(log2(prof[count].max_weight)));
		fprintf(fid, "%d %d %d %d %d %d %d %d %d %d\n", i, in_bw, in_fl, 
						out_bw, out_fl, w_bw, w_fl, mode, a_type, w_type);
		count++;
	}
	fclose(fid);
}
// TODO: replace with proper quantized cfg parsing
void read_quantized_net_cfg(network *net, char *filename) {
	FILE *fid = fopen(filename, "r");
	if(!fid) error("Unable to open file to read quantization cfg.");
	int size=1024;
	char buffer[size];
	const int num_params = 10;
	int params[num_params];
	char *head;
	int i,j,num;
	for (i=0; i<net->n; ++i){
		layer *l = &(net->layers[i]);
		if (!l->quantize)
			l->quantize = calloc(1,sizeof(quantize_params));
		quantize_params *q = l->quantize;
		if(l->weights){
			if(fgets(buffer, size, fid)==NULL){
				fprintf(stderr, "Unable to read necessary line from %s.\n", filename);
				exit(0);
			}
			head = buffer;
			num = 0;
			for(j=0; buffer[j] != '\0' && j < size-1; ++j){
				if(buffer[j]==' ' || buffer[j]==',' || buffer[j]=='\n'){
					buffer[j] = '\0';
					params[num++] = atoi(head);
					head = buffer+j+1; 
				}
				if(num >= num_params)
				 	 break;
			}
			assert(i==params[0]);
		}
		q->in_bw = params[1];
		q->in_fl = params[2];
		q->out_bw = params[3];
		q->out_fl = params[4];
		q->w_bw = params[5];
		q->w_fl = params[6];
		q->mode = params[7];
		q->a_type = params[8];
		q->w_type = params[9];
	}
	fclose(fid);
}


void print_net_weights(network *net, const char *filename){
	FILE *fid = fopen(filename, "w");
	if(!fid)
		error("cannot open file to print the weights");
	for (int i=0; i<net->n; ++i){
		layer *l = &(net->layers[i]);
		if (!l->weights)
			continue;
		int num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
		int num_biases = (l->nbiases > 0) ? l->nbiases : l->outputs;
#ifdef GPU
		if(gpu_index >= 0){
			cuda_pull_array(l->weights_gpu, l->weights, num_weights);
			cuda_pull_array(l->biases_gpu, l->biases, num_biases);
		}
#endif
		fprintf(fid,"Layer: %d, Type: %d\n", i, l->type);
		for (int j=0; j < num_weights; ++j){
			fprintf(fid,"%f ", l->weights[j]);
		}
		fprintf(fid,"\n");
		for (int j=0; j < num_biases; ++j){
			fprintf(fid,"%f ", l->biases[j]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}

void calc_TPFP_TNFN(network *net, int n, float *rate){

	if (!net->truth){
		fprintf(stderr, "Warning, no ground truth data for accuracy computation!\n");
		return;
	}
	float *output = net->output;
	float *truth = net->truth;
	int size = net->h*net->w;
	for (int i=0; i<net->batch; ++i){
		int TP=0, FP=0, TN=0, FN=0;		
		int offset = i*net->outputs + size;
		for (int j=offset; j < offset + size; ++j){
			if (output[j] > 0.5 && truth[j] > 0.5){
				TP += 1;
			}else if(output[j] > 0.5 && truth[j] < 0.5){
				FP += 1;
			}else if(output[j] < 0.5 && truth[j] > 0.5){
				FN += 1;
			}else{
				TN += 1;
			}
		}
		float precision = (float)TP/(TP+FP);
		float recall = (float)TP/(TP+FN);
		// compute Precision and Recall:
		rate[0] += precision;
		rate[1] += recall;
		// F-1 measure
		rate[2] += 2*(precision*recall)/(precision+recall);
		// E-1 accuracy rate
		rate[3] += (float)(FP+FN)/size;
		rate[4] += (float)(FP+FN)/size/2; 
	}
}


void run_and_calc_seg_accuracy(network *net, load_args *arguments, int N, float *results){		
    data train;
    data buffer;
    pthread_t load_thread;
	load_args args = *arguments;
    args.d = &buffer;
    load_thread = load_data(args);    	
	
	long int seen = *net->seen;
	*net->seen = 0;
	
	int epoch = 0;
    while(epoch == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

    	assert(train.X.rows % net->batch == 0);
		int batch = net->batch;
 		int n = train.X.rows / batch;
		net->train = 1;
    	for(int i = 0; i < n; ++i){
        	get_next_batch(train, batch, i*batch, net->input, net->truth);
	    	*net->seen += net->batch;
    		forward_network(net);
			// compute F-measures, Recall, Precision, E1 and E2:
			calc_TPFP_TNFN(net, net->batch*net->outputs, results);
		}
        free_data(train);	
		epoch = *net->seen/N;
    }
	
	assert(*net->seen > 0);
	for (int j=0; j<5; ++j){
		results[j] /= (float)(*net->seen);
	}
	*net->seen = seen;
}

void test_quantization(char *datacfg, char *cfgfile, char *weightfile, 
						int gpu, char *quantized_cfg, char *filename)
{
    int i;
#ifdef GPU
    if(gpu>=0) cuda_set_device(gpu);
#endif
    network *net = load_network(cfgfile, weightfile, 1);
	if (net->batch > 1)
		error("Not supporting testing with larger batch size at the moment");
	net->train = 0;	
	fold_batch_norm_params(net);
	// Initialize the quantization parameters:
	if (quantized_cfg) {
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
	}else{
		// RECALL, PRECISION, F1, E1, E2
		float acc[NUM_SEG_ACCURACY_ELEMENTS] = {0};
    	list *options = read_data_cfg(datacfg);
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
    	args.scale = 1;

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
    	args.n = net->batch*net->subdivisions;
    	args.m = N;
    	args.type = SEGMENTATION_DATA;

		run_and_calc_seg_accuracy(net, &args, N, acc);	
		fprintf(stderr, "Recall: %.4f, Precision: %.4f, F-1: %.4f, E1: %.4f, E2: %.4f\n",
				acc[0], acc[1], acc[2], acc[3], acc[4]);
		// test batch norm folding:
		//fold_batch_norm_params(net);
		//run_and_calc_seg_accuracy(net, &args, N, acc);	
		//fprintf(stderr, "Recall: %.4f, Precision: %.4f, F-1: %.4f, E1: %.4f, E2: %.4f\n",
		//		acc[0], acc[1], acc[2], acc[3], acc[4]);
		if(paths) free_ptrs((void**)paths, plist->size);
    	if(plist) free_list(plist);
	}
	
	for (i=0; i<net->n; ++i) {
		free_layer(net->layers[i]);
	}
	free(net->layers);
	if(net->truth) free(net->truth);
    free(net);
}
// this is mostly similar to segmenter_train
// TODO: make more generic and more data types than just SEGMENTATION_DATA
// TODO: support more than conv, deconv, and fc
void perform_quantization(int finetune, char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *quantized_cfg)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        if(gpus[i]>=0) cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);

        if(!finetune)
			set_batch_network(nets[i], 1);
		nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    image pred = get_network_image(net);

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

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
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

	// Initialize the quantization parameters:
	profiler *prof=NULL;
	if (!finetune) {
		fold_batch_norm_params(net);
		int num_prof_layer = 0;
		for (i=0; i < net->n; ++i) {
			if (!net->layers[i].weights)
				continue;
			num_prof_layer += 1;
		}
		prof = calloc(num_prof_layer, sizeof(profiler));
		net->max_batches = 100;
		net->train = 0;
	}else{
		allocate_quantized_weight_copy(net);
		read_quantized_net_cfg(net, quantized_cfg);
	}
		
    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);
	
	init_prune_mask(net, 0);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        double time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
        if (finetune){
    		loss = train_network_quantized(net, train); 
        }else{
			// just analyze the magnitudes:
			analyze_activations(net, train, prof);
			free_data(train);
			continue;
        }

        if (avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n",
        		get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss,
        		get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(finetune && *net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
    }
	
    char buff[256];
	if(finetune){	
    	sprintf(buff, "%s/%s.finetuned_weights", backup_directory, base);
    	save_weights(net, buff);
	}else{
		sprintf(buff, "%s/%s.quantized_cfg", backup_directory, base);
		write_quantized_net_cfg(net, prof, buff);
	}

	if(prof) free(prof);
	for (i=0; i<net->n; ++i) {
		free_layer(net->layers[i]);
	}
	free(net->layers);
	if(net->truth) free(net->truth);
    free(net);
    if(paths) free_ptrs((void**)paths, plist->size);
    if(plist) free_list(plist);
    if(base) free(base);
}

void run_quantizer(int argc, char **argv)
{
    if(argc < 4) {
        fprintf(stderr, "usage: %s %s [analyze/finetune] [cfg] [weights]\n", argv[0], argv[1]);
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
	int nq = find_arg(argc, argv, "-no-quantize-cfg");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;

	// adhoc implementation at the moment:
	// TODO: write new cfg file with proper quantization params
	char *quantized_cfg = (argc > 6) ? argv[6] : 0;
    char *filename = (argc > 7) ? argv[7]: 0;
	if (nq){
		filename = quantized_cfg;
		quantized_cfg = NULL;
	}
	
	if(0==strcmp(argv[2], "analyze")){
		if (gpu_index >= 0)
			error("Not supporting GPU execution for \"analyze\" at the moment");
		perform_quantization(0, data, cfg, weights, gpus, ngpus, 1, NULL);
	} else if(0==strcmp(argv[2], "finetune")) {
		if(!quantized_cfg) error("Must provide the quantization params file.");
		perform_quantization(1, data, cfg, weights, gpus, ngpus, clear, quantized_cfg);
	} else if(0==strcmp(argv[2], "test")) {
		test_quantization(data, cfg, weights, gpus[0], quantized_cfg, filename);
	}
	return;
}

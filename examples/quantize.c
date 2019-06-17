#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <math.h>

extern void quantize_cpu(float *x, int n, int bw, int fl, ROUNDING_MODE mode, QUANTIZATION_TYPE type);
#ifdef GPU
extern void quantize_gpu(float *x, int n, int bw, int fl, ROUNDING_MODE mode, QUANTIZATION_TYPE type);
#endif

// implment the selection algorithm
void selection(float *val, int n, int k){
	assert(k >= 0);
	assert(k <= n);
	int i, j;
	float maxVal;
	for (i=0; i < n; ++i)
		val[i] = fabs(val[i]);

	for (i=0; i <= k; ++i){
		maxVal = val[i];
		for (j=i+1; j < n; ++j){
			val[j] = val[j];
			if (val[j] > maxVal){
				maxVal = val[j];
				// swap
				val[j] = val[i];
				val[i] = maxVal;
			}
		}
	}
}

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
				selection(weight_copy, num_weights, k);
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
				selection(bias_copy, num_weights, k);
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
				selection(act_copy, num_acts, k);
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

#define BITWIDTH 8
#ifndef INT_MIN
#define INT_MIN -2147483648
#endif

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
		out_fl = BITWIDTH - (int)(ceil(log2(prof[count].val99_activation)));
		w_bw = BITWIDTH;
		w_fl = BITWIDTH - (int)(ceil(log2(prof[count].val99_weight)));
		fprintf(fid, "%d %d %d %d %d %d %d %d %d %d\n", i, in_bw, in_fl, 
						out_bw, out_fl, w_bw, w_fl, mode, a_type, w_type);
		count++;
	}
	fclose(fid);
}
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
		if(!net->layers[i].weights)
			continue;
		if(fgets(buffer, size, fid)==NULL)
			error("Unable to read necessary line from ordering.tmp.");
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
		quantize_params *q = net->layers[i].quantize;
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

void quantize_weights(network* net){
	int i;
	for (i=0; i<net->n; ++i){
		layer *l = &(net->layers[i]);
		if (!l->weights)
			continue;
		
		quantize_params *q = l->quantize;
		int num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
		int num_biases = (l->nbiases > 0) ? l->nbiases : l->outputs;

#ifdef GPU
		if (gpu_index >= 0){
			cudaError_t status;
    		status = cudaMemcpy(l->weights_gpu, q->weight_copy_gpu, num_weights*sizeof(float), cudaMemcpyDeviceToDevice);
    		//check_error(status);
    		status = cudaMemcpy(l->biases_gpu, q->bias_copy_gpu, num_biases*sizeof(float), cudaMemcpyDeviceToDevice);
    		//check_error(status);

			quantize_gpu(l->weights_gpu, num_weights, q->w_bw, q->w_fl, q->mode, q->w_type);
			quantize_gpu(l->biases_gpu, num_biases, q->out_bw, q->out_fl, q->mode, q->a_type);
			continue;
		}
#endif

		memcpy(l->weights, q->weight_copy, num_weights*sizeof(float));
		memcpy(l->biases, q->bias_copy, num_biases*sizeof(float));
		quantize_cpu(l->weights, num_weights, q->w_bw, q->w_fl, q->mode, q->w_type);
		quantize_cpu(l->biases, num_biases, q->out_bw, q->out_fl, q->mode, q->a_type);
	}
}

void swap_quantized_weight_pointers(network *net){
	float *w, *b;
	for (int i=0; i<net->n; ++i){
		layer *l = &(net->layers[i]);
		if (!l->weights)
			continue;

#ifdef GPU
		if (gpu_index >= 0){
			w = l->weights_gpu;
			b = l->biases_gpu;

			l->weights_gpu = l->quantize->weight_copy_gpu;
			l->biases_gpu = l->quantize->bias_copy_gpu;
		
			l->quantize->weight_copy_gpu = w;
			l->quantize->bias_copy_gpu = b;
			continue;
		}
#endif

		w = l->weights;
		b = l->biases;
		
		l->weights =  l->quantize->weight_copy;
		l->biases = l->quantize->bias_copy;

		l->quantize->weight_copy = w;
		l->quantize->bias_copy = b;
	}
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

void test_quantization(char *datacfg, char *cfgfile, char *weightfile, int gpu, char *quantized_cfg, char *filename)
{
    int i;
#ifdef GPU
    if(gpu>=0) cuda_set_device(gpu);
#endif
    network *net = load_network(cfgfile, weightfile, 1);
	net->train = 0;
	set_batch_network(net, 1);
	
	// Initialize the quantization parameters:
	if (quantized_cfg) {
		for (i=0; i < net->n; ++i) {
			layer *l = &(net->layers[i]);
			if (!l->weights)
				continue;
			l->quantize = calloc(1,sizeof(quantize_params));
		}
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

    	data train;
    	data buffer;
    	pthread_t load_thread;
    	args.d = &buffer;
    	load_thread = load_data(args);
		
    	int epoch = 0;
    	while(epoch == 0){
        	double time = what_time_is_it_now();
        	pthread_join(load_thread, 0);
        	train = buffer;
        	load_thread = load_data(args);

        	printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        	time = what_time_is_it_now();

    		assert(train.X.rows % net->batch == 0);
			int batch = net->batch;
 			int n = train.X.rows / batch;
			net->train = 1;
    		for(i = 0; i < n; ++i){
        		get_next_batch(train, batch, i*batch, net->input, net->truth);
	    		*net->seen += net->batch;
    			forward_network(net);
			}
        	free_data(train);
        	
			epoch = *net->seen/N;
    	}
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
    int num_prof_layer = 0;
	profiler *prof=NULL;

	for (i=0; i < net->n; ++i) {
		layer *l = &(net->layers[i]);
		if (!l->weights)
			continue;
		int num_weights, num_biases;
		// true for conv, deconv and fc layers.
		num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
		num_biases = (l->nweights > 0) ? l->n : l->outputs;
		assert(num_weights > 0);

		if (finetune) {
			l->quantize = calloc(1,sizeof(quantize_params));
			l->quantize->weight_copy = calloc(num_weights, sizeof(float));
			l->quantize->bias_copy = calloc(num_biases, sizeof(float));

#ifdef GPU
			if (gpu_index >= 0){
				l->quantize->weight_copy_gpu = cuda_make_array(l->weights, num_weights);
				l->quantize->bias_copy_gpu = cuda_make_array(l->biases, num_biases);
			}
#endif
			// copy over the original weights:
			memcpy(l->quantize->weight_copy, l->weights, num_weights*sizeof(float));
			memcpy(l->quantize->bias_copy, l->biases, num_biases*sizeof(float));
		}
		num_prof_layer += 1;
	}
	if (!finetune) {
		prof = calloc(num_prof_layer, sizeof(profiler));
		net->max_batches = 50;
		net->train = 0;
	}
	//net->max_batches = 1;
		
	if (quantized_cfg) read_quantized_net_cfg(net, quantized_cfg);

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
        if (finetune){
    		assert(train.X.rows % net->batch == 0);
			int batch = net->batch;
 			int n = train.X.rows / batch;

			float sum = 0;
			net->train = 1;
    		for(i = 0; i < n; ++i){
        		get_next_batch(train, batch, i*batch, net->input, net->truth);
	    		*net->seen += net->batch;
				// synchronize and quantize weights:
				quantize_weights(net);

    			forward_network(net);

    			// swap pointers for quantized and backup weights:
				swap_quantized_weight_pointers(net);
				
				//print_net_weights(net, "normal_weights.tmp");

				backward_network(net);
				sum += *net->cost;
				if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);

				//print_net_weights(net, "updated_weights.tmp");
    			// swap back:
				swap_quantized_weight_pointers(net);
				//print_net_weights(net, "quantized_weights.tmp");
    		}
    		loss = (float)sum/(n*batch);
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
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;

	// adhoc implementation at the moment:
	// TODO: write new cfg file with proper quantization params
	char *quantize_info = (argc > 6) ? argv[6] : 0;
    char *filename = (argc > 7) ? argv[7]: 0;
    if(0==strcmp(argv[2], "analyze")){
		perform_quantization(0, data, cfg, weights, gpus, ngpus, 1, NULL);
	} else if(0==strcmp(argv[2], "finetune")) {
		if(!quantize_info) error("Must provide the quantization params file.");
		perform_quantization(1, data, cfg, weights, gpus, ngpus, clear, quantize_info);
	} else if(0==strcmp(argv[2], "test")) {
		if(!quantize_info) error("Must provide the quantization params file.");
		test_quantization(data, cfg, weights, gpus[0], quantize_info, filename);
	}
	return;
}

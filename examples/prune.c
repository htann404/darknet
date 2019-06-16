#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <math.h>

// Make this layer adaptive:
#define THRESH 0.02

// Implement iterative pruning:
void prune(network *net){
	int n = net->n;
	int i, j, num_weights;
	float w;
	for (i=0; i<n; ++i){
		layer l = net->layers[i];
		if(!l.weights)
			continue;	
		// true for conv, deconv and fc layers.
		num_weights = (l.nweights > 0) ? l.nweights : l.inputs*l.outputs;
		assert(num_weights > 0);
#ifdef GPU
		if(gpu_index >= 0){
			cuda_pull_array(l.weights_gpu, l.weights, num_weights);
			cuda_pull_array(l.weight_prune_mask_gpu, l.weight_prune_mask, num_weights);
		}
#endif
		for (j=0; j<num_weights; ++j){
			w = fabs(l.weights[j]);
			if (w < THRESH){
				l.weights[j] = 0;
				l.weight_prune_mask[j] = 0;
			}
		}
#ifdef GPU
		if(gpu_index >= 0){
			cuda_push_array(l.weights_gpu, l.weights, num_weights);
			cuda_push_array(l.weight_prune_mask_gpu, l.weight_prune_mask, num_weights);
		}
#endif	
	}
}


// this is mostly similar to segmenter_train
// TODO: make more generic and more data types than just SEGMENTATION_DATA
// TODO: support more than conv, deconv, and fc
void iterative_pruning(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i,j;

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
    image pred = get_network_image(net);

    int div = net->w/pred.w;
    assert(pred.w * div == net->w);
    assert(pred.h * div == net->h);

    int imgs = net->batch * net->subdivisions * ngpus;

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
    args.n = imgs;
    args.m = N;
    args.type = SEGMENTATION_DATA;

	// Initialize the prune mask:
	for (i=0; i<net->n; ++i) {
		layer *l = &(net->layers[i]);
		if (!l->weights)
			continue;
		int num_weights;
		// true for conv, deconv and fc layers.
		num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
		assert(num_weights > 0);
		l->weight_prune_mask = malloc(num_weights*sizeof(float));
		for (j=0; j<num_weights; ++j){
			l->weight_prune_mask[j] = 1;
		}
#ifdef GPU
		if (gpu_index >= 0){
			l->weight_prune_mask_gpu = cuda_make_array(l->weight_prune_mask, num_weights);
		}
#endif
	}
	prune(net);

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
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
			prune(net);

            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.pruned_weights", backup_directory, base);
    save_weights(net, buff);
	
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

void count_zeros(char *datacfg, char *cfgfile, char *weightfile)
{
    int i,j,count,num_weights;
	network *net = load_network(cfgfile, weightfile, 0);

	for (i=0; i<net->n; ++i) {
		count = 0;
		layer *l = &(net->layers[i]);
		if(!l->weights)
			continue;
		num_weights = (l->nweights > 0) ? l->nweights : l->inputs*l->outputs;
		for (j=0; j<num_weights; ++j)
			if (l->weights[j]==0)
				++count;
		fprintf(stderr, "Layer %d, Type %d, Num Weights: %d, Fraction Zeros: %.4f\n",
				i, l->type, num_weights, ((float)count)/((float)num_weights) );
	}

	free(net);
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
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    //char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "count")){
		count_zeros(data, cfg, weights);
	}
    else if(0==strcmp(argv[2], "iterative")) iterative_pruning(data, cfg, weights, gpus, ngpus, clear);
}



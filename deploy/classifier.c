#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

void validate_classifier_from_net(network net_in, int topk, 
                                  load_args args, float *results)
{
    int i, j;
    network *net = &net_in;
    set_batch_network(net, 1);

    char **labels = args.labels;
    char **paths = args.paths;
    int classes = args.classes;
    int m = args.m;

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred;
        if (net->true_q) {
            image_Dtype imD, cropD;
            imD = load_image_color_Dtype(paths[i], 0, 0);
            cropD = center_crop_image_Dtype(imD, net->w, net->h);
            pred = network_predict_Dtype(net, cropD.data);
            free_image_Dtype(imD);
            free_image_Dtype(cropD);
        }else{
            image im, crop;
            im = load_image_color(paths[i], 0, 0);
            crop = center_crop_image(im, net->w, net->h);
            pred = network_predict(net, crop.data);
            free_image(im);
            free_image(crop);
        }
        // not implemented
        //if(net->hierarchy)
              //hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }
    }
    free(indexes);
    results[0] = avg_acc/m; results[1] = avg_topk/m;
    printf("%d: top 1: %f, top %d: %f\n", i, results[0], topk, results[1]);
}

void test_classifier(int q, int topk, char *datacfg, char *cfgfile, 
              char *weightfile, char *quantized_cfg, char *filename)
{
    int i;
    network *net = load_network(cfgfile, weightfile, 1);
    if (net->batch > 1){
        set_batch_network(net, 1);
        fprintf(stderr, "Batch size set to 1.\n");
    }

    net->train = 0;    
    // Initialize the quantization parameters:
    if (quantized_cfg){
        read_quantized_net_cfg(net, quantized_cfg);
        fprintf(stderr, "Done reading quantization config file.\n");
        if(q){
            net->true_q = 1;
            fprintf(stderr, "Performing inference using true quantized datapaths.\n");
        }
    }else{
        fprintf(stderr, "Warning! Quantization config is not provided.\n");
    }

    if (filename){
        image im, sized;
        image_Dtype imD, sizedD;
        float *predictions;
        double time;
        if (net->true_q){
            imD = load_image_color_Dtype(filename,0,0);
            sizedD = center_crop_image_Dtype(imD, net->w, net->h);
            time = clock();
            predictions = network_predict_Dtype(net, sizedD.data);
        }else{
            image im = load_image_color(filename, 0, 0);
            image sized = center_crop_image(im, net->w, net->h);
            time = clock();
            predictions = network_predict(net, sized.data);        
        }
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        printf("Predicted: %f\n", predictions[0]);
        if (net->true_q){
            free_image_Dtype(imD);
            free_image_Dtype(sizedD);
        }else{
            free_image(im);
            free_image(sized);
        }
    }else{
        // top 1 and top k
        float acc[2] = {0};
        list *options = read_data_cfg(datacfg);
        char *train_list = option_find_str(options, "valid", "data/valid.list");
        list *plist = get_paths(train_list);

        char **paths = (char **)list_to_array(plist);
        printf("Number of images: %d\n", plist->size);
        int N = plist->size;
        load_args args = set_load_args(CLASSIFICATION, net, paths, options, N);
        validate_classifier_from_net(*net, topk, args, acc);

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

void run_classifier(int argc, char **argv)
{
    if(argc < 4) {
        fprintf(stderr, "usage: %s %s test [data_cfg] [cfg] [weights]\n", argv[0], argv[1]);
        return;
    }

    int nq = find_arg(argc, argv, "-no-quantize-cfg");
    int true_q = find_arg(argc, argv, "-true-quantize");
    int top = find_int_arg(argc, argv, "-t", 0);
    // default to classification

    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;

    // adhoc implementation at the moment:
    // TODO: write new cfg file with quantization params included
    char *quantized_cfg = (argc > 6) ? argv[6] : 0;
    char *filename = (argc > 7) ? argv[7]: 0;
    if (nq){
        true_q = 0;
        filename = quantized_cfg;
        quantized_cfg = NULL;
    }
    
    test_classifier(true_q, top, data, cfg, weights, quantized_cfg, filename);
    return;
}


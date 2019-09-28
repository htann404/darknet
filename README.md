# Quarknet #
Quarknet is an extension of the [Darknet](https://github.com/pjreddie/darknet) framework. Added features include supports for model pruning and quantization. A truely quantized and sparse execution is also supported for 8-bit dynamic fixed-point weights and activations. For more information, see our [technical report on Arxiv](arxiv.org).

## Installation
Installation instructions and tutorials on how to use Darknet can be found at the [Darknet project website](https://pjreddie.com/darknet/).

## Quarknet extension
The extension introduced in Quarknet is targeted for classification and segmentation tasks. More specifically, only convolution, deconvolution and fully connected layer types are supported for pruning and quantization. The framework can easily be extended to other layer types.

### CIFAR-10 example
Below is an examples 

## License and Citation
License for Darknet can be found at its [repository](https://github.com/pjreddie/darknet).

Quarknet extension is released under the [BSD 2-Clause license](https://github.com/scale-lab/darknet/blob/master/LICENSE.quarknet).

If you use Quarknet, please cite our paper:
......

# SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
<br>

<p align="center">
<img src="images/sde_animation.gif" width="320"/>
</p>

[**Project**](https://sde-image-editing.github.io/) | [**Paper**](https://arxiv.org/abs/2108.01073) | [**Colab**](https://colab.research.google.com/drive/1KkLS53PndXKQpPlS1iK-k1nRQYmlb4aO?usp=sharing)

PyTorch implementation of **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** (ICLR 2022).

[Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Yutong He](http://web.stanford.edu/~kellyyhe/), [Yang Song](https://yang-song.github.io/), [Jiaming Song](http://tsong.me/),
[Jiajun Wu](https://jiajunwu.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Stefano Ermon](https://cs.stanford.edu/~ermon/)

Stanford and CMU


<p align="center">
<img src="images/teaser.jpg" />
</p>

Recently, SDEdit has also been applied to text-guided image editing with large-scale text-to-image models. Notable examples include <a href="https://en.wikipedia.org/wiki/Stable_Diffusion">Stable Diffusion</a>'s img2img function (see  <a href="https://github.com/CompVis/stable-diffusion#image-modification-with-stable-diffusion">here</a>), <a href="https://arxiv.org/abs/2112.10741">GLIDE</a>, and <a href="https://arxiv.org/abs/2210.03142">distilled-SD</a>. The below example comes from <a href="https://arxiv.org/abs/2210.03142">distilled-SD</a>.

<p align="center">
<img src="images/text_guided_img2img.png" />
</p>


## Overview
The key intuition of SDEdit is to "hijack" the reverse stochastic process of SDE-based generative models, as illustrated in the figure below. Given an input image for editing, such as a stroke painting or an image with color strokes, we can add a suitable amount of noise to make its artifacts undetectable, while still preserving the overall structure of the image. We then initialize the reverse SDE with this noisy input, and simulate the reverse process to obtain a denoised image of high quality. The final output is realistic while resembling the overall image structure of the input.

<p align="center">
<img src="images/sde_stroke_generation.jpg" />
</p>

## Getting Started
The code will automatically download pretrained SDE (VP) PyTorch models on
[CelebA-HQ](https://huggingface.co/XUXR/SDEdit/resolve/main/celeba_hq.ckpt),
[LSUN bedroom](https://huggingface.co/XUXR/SDEdit/blob/main/celeba_hq.ckpt), 
and [LSUN church outdoor](https://huggingface.co/XUXR/SDEdit/blob/main/ema_lsun_church.ckpt).

### Data format
We save the image and the corresponding mask in an array format ``[image, mask]``, where
"image" is the image with range ``[0,1]`` in the PyTorch tensor format, "mask" is the corresponding binary mask (also the PyTorch tensor format) specifying the editing region.
We provide a few examples, and ``functions/process_data.py``  will automatically download the examples to the ``colab_demo`` folder.

### Re-training the model
Here is the [PyTorch implementation](https://github.com/ermongroup/ddim) for training the model.


## Stroke-based image generation
Given an input stroke painting, our goal is to generate a realistic image that shares the same structure as the input painting.
SDEdit can synthesize multiple diverse outputs for each input on LSUN bedroom, LSUN church and CelebA-HQ datasets.



To generate results on LSUN datasets, please run

```
python main.py --exp ./runs/  --config celeba.yml --img <path_to_img.jpg> --sample -i images --sample_step 3 --t 500  --ni
```
```
python main.py --exp ./runs/ --config church.yml --sample -i images --img <path_to_img.jpg> --sample_step 3 --t 500  --ni
```

Use all images in a directory, please run

```
python main.py --exp ./runs/ --config church.yml --sample -i images --sample_step 1 --t 300  --ni --init_dir <path to dir of images>
```
<p align="center">
<img src="images/stroke_based_generation.jpg" width="800">
</p>

## Stroke-based image editing
Given an input image with user strokes, we want to manipulate a natural input image based on the user's edit.
SDEdit can generate image edits that are both realistic and faithful (to the user edit), while avoid introducing undesired changes.
<p align="center">
<img src="images/stroke_edit.jpg" width="800">
</p>

To perform stroke-based image editing, run

```
python main.py --exp ./runs/  --config church.yml --sample -i images --img <path to image> --sample_step 3 --t 500  --ni
```

## Additional results
<p align="center">
<img src="images/stroke_generation_extra.jpg" width="800">
</p>

## References
If you find this repository useful for your research, please cite the following work.
```
@inproceedings{
      meng2022sdedit,
      title={{SDE}dit: Guided Image Synthesis and Editing with Stochastic Differential Equations},
      author={Chenlin Meng and Yutong He and Yang Song and Jiaming Song and Jiajun Wu and Jun-Yan Zhu and Stefano Ermon},
      booktitle={International Conference on Learning Representations},
      year={2022},
}
```

This implementation is based on / inspired by:

- [DDIM PyTorch repo](https://github.com/ermongroup/ddim).
- [DDPM TensorFlow repo](https://github.com/hojonathanho/diffusion).
- [PyTorch helper that loads the DDPM model](https://github.com/pesser/pytorch_diffusion).
- [code structure](https://github.com/ermongroup/ncsnv2).

Here are also some of the interesting follow-up works of SDEdit:

- [Image Modification with Stable Diffusion](https://github.com/CompVis/stable-diffusion#image-modification-with-stable-diffusion)

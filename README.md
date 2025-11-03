# Diffusion Models

Trying out [Diffusion](https://arxiv.org/abs/2006.11239) and [Latent Diffusion](https://arxiv.org/abs/2112.10752) Models.

### Dataset & Preprocessing

|Dataset|Image Size|Description|
|:---:|:---:|:---:|
|Fashion-MNIST|$28\times 28$ grayscale|MNIST of clothing items|
|CelebA-HQ|$64\times 64$ RGB|High-quality celebrity faces|

- Preprocessing: Normalized to [-1, 1]
- Data augmentation: Random Horizontal Flip
- Training on T4 GPU (Google Colab)

### Evaluation Metric: Fréchet Inception Distance (FID)

FID measures similarity between real and generated images in feature space.
Lower scores indicate closer resemblance to real data.

```bash
python -m pytorch_fid /path/to/real_images /path/to/generated_images
```

> Big thanks to [Phil Wang's implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) and the writers of [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) for their wonderful article on DDPM.

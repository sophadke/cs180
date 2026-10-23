# CS180 Project 3 — Flow Models

Completed Part A and Part B code and website results.

## Website

- [Portfolio index](../index.html) links directly to both parts.
- [Part A: Fun with Flow Models](project3-parta.html)
- [Part B: Flow Matching from Scratch](project3-partb.html)

The pages include implementations, figures, experimental settings, observations, and findings. Shared portfolio styling is in `../style.css`; Project 3 additions are in `parta.css`.

## Code

- **Combined submission:** [`notebooks/CS180_Project3_Combined.ipynb`](notebooks/CS180_Project3_Combined.ipynb)
- [Part A notebook](notebooks/parta.ipynb)
- [Part B notebook](notebooks/partb.ipynb)

Open the combined notebook in Colab with a GPU. Its Part A setup embeds the support code and input images; Part B downloads MNIST. The transition cell clears Part A's variables and releases GPU memory before Part B starts. Each part can also be run separately in a fresh GPU runtime.

Part A uses pretrained PixNerd for sampling and editing. Part B trains five independent MNIST experiments: single-step denoising, pure-noise denoising, time-conditioned flow matching, class-conditioned flow matching, and a constant-learning-rate comparison.

## Saved results

`results/partA/` and `results/partB/` contain the website figures and measured run settings. All five Part B experiments completed; its saved figures are also included in the Part B and combined notebooks. Part A's source notebook did not retain execution outputs, so its completed figures are provided on the website.

An earlier hosted runtime expired. The denoising and time-only figures and epoch summaries were recovered from notebook outputs; their original raw per-step arrays and model weights were unavailable. Class-conditioned raw losses were preserved in Part B's metrics. Large checkpoint archives are kept locally rather than included in this website repository.

## Conventions

- Part A: RGB in `[-1,1]`; noise at time 0, clean image at time 1; guidance `vc + w*(vc-vu)`.
- Part B: grayscale in `[0,1]`; noise at time 0, data at time 1; guidance `vu + gamma*(vc-vu)`.

[Official assignment](https://cal-cs180.github.io/fa26/hw/proj3-flow/index.html). This project follows the CS180 scope.

# Introduction to Score-based Generative Modeling

## Task 0: Introduction
We know that a stochastic differential equation has the following form:
$$d\mathbf{X}_t = f(t,\mathbf{X}_t)dt + G(t)d\mathbf{B}_t$$
where $f$ and $G$ are the drift and diffusion coefficients respectively and $\mathbf{B}_t$ is the
standard Brownian noise. A popular SDE often used is called Ornstein-Uhlenbeck (OU) process
which is defined as
$$d\mathbf{X}_t = \mu \mathbf{X}_tdt + \sigma d\mathbf{B}_t$$
Where $\mu, \sigma$ are constants. In this tutorial, we will set $\mu = \frac{1}{2}, \sigma = 1$.
Score-based generative modelling (SGM) aims to sample from an unknown distribution of a given dataset.
We have the following two observations:
- The OU process always results in a unit Gaussian.
- We can derive the equation for the inverse OU process.

From these facts, we can directly sample from the unknown distribution by
1. Sample from unit Gaussian
2. Run the reverse process on samples from step 1.

[Yang Song et al. (2021)](https://arxiv.org/abs/2011.13456) derived the likelihood training scheme
for learning the reverse process. In summary, the reverse process for any SDE given above is
of the form
$$d\mathbf{X}_t = [f(t,\mathbf{X}_t)dt - G(t)^2\nabla_x\log p_t(\mathbf{X}_t)] + G(t)d\bar{\mathbf{B}}_t$$
where $\bar{\mathbf{B}}_t$ is the reverse brownian noise. The only unknown term is the score function
$\nabla_x\log p_t(\mathbf{X}_t)$, which we will approximate with a Neural Network. One main difference
between SGM and other generative models is that they generate iteratively during the sampling process.

TODO:
- Derive close-form equation for the mean and std of OU process at time t.

## Task 1: Implement a simple pipeline for SGM with delicious swiss-roll
A typical diffusion pipeline is divided into three components:
1. Forward Process and Reverse Process
2. Training
3. Sampling

In this task, we will look into each component one by one and implement them sequentially.

## Resources
- [lecture] [Videos on Introduction to Diffusion](https://www.youtube.com/@quantpie/videos)
- [lecture]
- [lecture]
- [paper]
-

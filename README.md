<div align=center>
  <h1>
    Introduction to Score-based Generative Modeling
  </h1>
  <div align=center>
    <div align=center>
      <b>Nguyen Minh Hieu &emsp; Juil Koo</b>
    </div>
    <div align=center>
      <p align=center>{hieuristics, 63days} [at] kaist.ac.kr</p>
    </div>
  </div>
</div>

<div align=center>
   <img src="https://github.com/min-hieu/HelloScore/assets/53557912/6b07ea99-9936-47f3-ac40-3e432a013c88">
</div>

<details>
<summary>Table of Content</summary>

- [Task 0](#task-0-introduction)
- [Task 1](#task-1-very-simple-sgm-pipeline-with-delicious-swiss-roll)
  - [Task 1.1](#1-forward-and-reverse-process) [(a)](#a-ou-process), [(b)](#b-vpsde--vesde)
  - [Task 1.2](#2-training)
  - [Task 1.3](#3-sampling)
  - [Task 1.4](#4-evaluation)
- [Task 2](#task-2-image-diffusion)

</details>

## Setup

Install the required package within the `requirements.txt`
```
pip install -r requirements.txt
```

## Code Structure
```
.
├── image_diffusion (Task 2)
│   ├── dataset.py                <--- Ready-to-use AFHQ dataset code
│   ├── train.py                  <--- DDPM training code
│   ├── sampling.py               <--- Image sampling code
│   ├── ddpm.py                   <--- DDPM high-level wrapper code
│   ├── module.py                 <--- Basic modules of a noise prediction network
│   ├── network.py                <--- (TODO) Define a network architecture
│   └── scheduler.py              <--- (TODO) Define various variance schedulers
└── sde_todo        (Task 1)
    ├── HelloScore.ipynb          <--- main code
    ├── dataset.py                <--- Define dataset (Swiss-roll, moon, gaussians, etc.)
    ├── loss.py                   <--- (TODO) Define Training Objective
    ├── network.py                <--- (TODO) Define Network Architecture
    ├── sampling.py               <--- (TODO) Define Discretization and Sampling
    ├── sde.py                    <--- (TODO) Define SDE Processes
    └── train.py                  <--- (TODO) Define Training Loop
```

## Tutorial Tips

Implementation of Diffusion Models is typically very simple once you understand the theory.
So, to learn the most from this tutorial, it's highly recommended to check out the details in the
related papers and understand the equations **BEFORE** you start the tutorial. You can check out
the resources in this order:
1. [[blog](https://min-hieu.github.io/blogs/blogs/brownian/)] Charlie's "Brownian Motion and SDE"
2. [[paper](https://arxiv.org/abs/2011.13456)] Score-Based Generative Modeling through Stochastic Differential Equations
3. [[blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)] Lilian Wang's "What is Diffusion Model?"
4. [[paper](https://arxiv.org/abs/2006.11239)] Denoising Diffusion Probabilistic Models

## Task 0: Introduction
We know that a stochastic differential equation has the following form:
$$d\mathbf{X}_t = f(t,\mathbf{X}_t)dt + G(t)d\mathbf{B}_t$$
where $f$ and $G$ are the drift and diffusion coefficients respectively and $\mathbf{B}_t$ is the
standard Brownian noise. A popular SDE often used is called Ornstein-Uhlenbeck (OU) process
which is defined as
$$d\mathbf{X}_t = -\mu \mathbf{X}_tdt + \sigma d\mathbf{B}_t$$
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
$$d\mathbf{X}_t = [f(t,\mathbf{X}_t) - G(t)^2\nabla_x\log p_t(\mathbf{X}_t)]dt + G(t)d\bar{\mathbf{B}}_t$$
where $\bar{\mathbf{B}}_t$ is the reverse brownian noise. The only unknown term is the score function
$\nabla_x\log p_t(\mathbf{X}_t)$, which we will approximate with a Neural Network. One main difference
between SGM and other generative models is that they generate iteratively during the sampling process.

**TODO:**
```
- Derive the expression for the mean and std of the OU process at time t given X0 = 0,
  i.e. Find E[Xt|X0] and Var[Xt|X0]. You will need this for task 1.1(a).
```
*hint*: We know that the solution to the OU process is given as

$\mathbf{X}_T = \mathbf{X}_0 e^{-\mu T} + \sigma \int_0^T e^{-\mu(T-t)} d\mathbf{B}_t$

and you can use the fact that $d\mathbf{B}_t^2 = dt$, and $\mathbb{E}[\int_0^T f(t) d\mathbf{B}_t] = 0$ where $f(t)$ is any
deterministic function.

## Task 1: very simple SGM pipeline with delicious swiss-roll
A typical diffusion pipeline is divided into three components:
1. [Forward Process and Reverse Process](#1-forward-and-reverse-process)
2. [Training](#2-training)
3. [Sampling](#3-sampling)

In this task, we will look into each component one by one and implement them sequentially.
#### 1. Forward and Reverse Process
<p align="center">
<img width="364" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/70352400-ac68-4509-b648-f64adcc602bc">
</p>

Our first goal is to setup the forward and reverse processes. In the forward process, the final distribution should be
the prior distribution which is the standard normal distribution.
#### (a) OU Process
Following the formulation of the OU Process introduced in the previous section, complete the `TODO` in the
`sde.py` and check if the final distribution approach unit gaussian as $t\rightarrow \infty$.

<p align="center">
<img width="840" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/d7d9341f-bff9-471c-b8e8-922fcddb8c09">
</p>


**TODO:**
```
- implement the forward process using the given marginal probability p_t0(Xt|X0) in SDE.py
- implement the reverse process for general SDE in SDE.py
- (optional) Play around with terminal time (T) and number of time steps (N) and observe its effect
```

#### (b) VPSDE & VESDE
It's mentioned by [Yang Song et al. (2021)](https://arxiv.org/abs/2011.13456) that the DDPM and SMLD are distretization of SDEs.
Implement this in the `sde.py` and check their mean and and std.

*hint*: Although you can simulate the diffusion process through discretization, sampling with the explicit equation of the marginal probability $p_{t0}(\mathbf{X}_t \mid \mathbf{X}_0)$ is much faster.

You should also obtain the following graphs for VPSDE and VESDE respectively
<p align="center">
  <img width="840" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/c7c5a3f1-675f-4817-8aa2-5d041c939ff6">
  <img width="840" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/bfd738a9-d562-4804-b982-5134b1e6884a">
">
</p>

**TODO:**
```
- implement VPSDE in SDE.py
- implement VESDE in SDE.py
- plot the mean and variance of VPSDE and VESDE vs. time.
  What can you say about the differences between OU, VPSDE, VESDE?
```

#### 2. Training
The typical training objective of diffusion model uses **D**enoising **S**core **M**atching loss:

$$f_{\theta^*} = \textrm{ argmin }  \mathbb{E} [||f_\theta(\mathbf{X}s) - \nabla p_{s0}(\mathbf{X}_s\mid \mathbf{X}_0)||^2] $$

Where $f$ is the score prediction network with parameter $\theta^*$.
Another popular training objective is **I**mplicit **S**core **M**atching loss which can be derived from DSM.
One main different between ISM and DSM is that ISM doesn't require marignal density but instead the divergence.
Although DSM is easier to implement, when the given [domain of interest](https://arxiv.org/abs/2202.02763) or
when the marginal density [doesn't have closed form](https://openreview.net/pdf?id=nioAdKCEdXB) ISM is used.

**Important** you need to derive a **different DSM object** for each SDE since
their marginal density is different. You first need to obtain the $p_{0t}(\mathbf{X}_t\mid \mathbf{X}_0$, then
find the equation for $\nabla \log p_{0t}(\mathbf{X}_t\mid \mathbf{X}_0$.

However, there are other training objectives with their different trade-offs (SSM, EDM, etc.). Highly recommend to checkout
[A Variational Perspective on Diffusion-based Generative Models and Score Matching](https://arxiv.org/abs/2106.02808)
and [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) for a more in-depth analysis of the recent training objectives.

**TODO:**
```
- implement your own network in network.py
  (Recommend to implement Positional Encoding, Residual Connection)
- implement ISMLoss in loss.py (hint: you will need to use torch.autograd.grad)
- implement DSMLoss in loss.py
- implement the training loop in train_utils.py
- (optional) implement SSMLoss in loss.py
```
#### 3. Sampling
Finally, we can now use the trained score prediction network to sample from the swiss-roll dataset. Unlike the forward process, there is no analytical form
of the marginal probabillity. Therefore, we have to run the simulation process. Your final sampling should be close to the target distribution
**within 10000 training steps**. For this task, you are free to use **ANY** variations of diffusion process that **was mentioned** above.

<p align="center">
  <img height="300" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/bb246de4-431c-4f0c-95ca-6f8323803e2c">
</p>


**TODO:**
```
- implement the predict_fn in sde.py
- complete the code in sampling.py
- (optional) train with ema
- (optional) implement the correct_fn (for VPSDE, VESDE) in sde.py
- (optional) implement the ODE discretization and check out their differences
```

#### 4. Evaluation
To evaluate your performance, we compute the chamfer distance (CD) and earth mover distance (EMD) between the target and generated point cloud.
Your method should be on par or better than the following metrics. For this task, you can use **ANY** variations, even ones that were **NOT** mentioned.

| target distribution |    CD    |
|---------------------|----------|
| swiss-roll          |  0.1975  |

#### 5. [Coming Soon] Schrödinger Bridge (Optional)
One restriction to the typical diffusion processes are that they requires the prior to be easy to sample (gaussian, uniform, etc.).
Schrödinger Bridge removes this limitation by making the forward process also learnable and allow a diffusion defined between **two** unknown distribution.
For example, with Schrödinger Bridge, you don't need to know the underlying distribution of the Moon dataset but you can still
define a diffusion (bridge) that map between the Moon dataset and the swiss roll as shown below.

Like any diffusion process, there are many ways to learn the Schrödinger Bridge. This section focus on the work presented in

## Task 2: Image Diffusion
In this task, we will play with diffusion models to generate 2D images. We first look into some background of DDPM and then dive into DDPM in a code level.

### Background
From the perspective of SDE, SGM and DDPM are the same models with only different parameterizations. As there are forward and reverse processes in SGM, the forward process, or called _diffusion process_, of DDPM is fixed to a Markov chain that gradually adds Gaussian noise to the data:

$$ q(\mathbf{x}\_{1:T} | \mathbf{x}_0) := \prod\_{t=1}^T q(\mathbf{x}_t | \mathbf{x}\_{t-1}), \quad q(\mathbf{x}_t | \mathbf{x}\_{t-1}) := \mathcal{N} (\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}\_{t-1}, \beta_t \mathbf{I}).$$


Thanks to a nice property of a Gaussian distribution, one can sample $\mathbf{x}_t$ at an arbitrary timestep $t$ from real data $\mathbf{x}_0$ in closed form:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I}) $$

where $\alpha\_t := 1 - \beta\_t$ and $\bar{\alpha}_t := \prod$ $\_{s=1}^T \alpha_s$.

Given the diffusion process, we want to model the _reverse process_ that gradually denoises white Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ to sample real data. It is also defined as a Markov chain with learned Gaussian transitions:

$$p\_\theta(\mathbf{x}\_{0:T}) := p(\mathbf{x}_T) \prod\_{t=1}^T p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}_t), \quad p\_\theta(\mathbf{x}\_{t-1} | \mathbf{x}_t) := \mathcal{N}(\mathbf{x}\_{t-1}; \mathbf{\boldsymbol{\mu}}\_\theta (\mathbf{x}_t, t), \boldsymbol{\Sigma}\_\theta (\mathbf{x}_t, t)).$$

To learn this reverse process, we set an objective function that minimizes KL divergence between $p_\theta(\mathbf{x}\_{t-1} | \mathbf{x}_t)$ and $q(\mathbf{x}\_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ which is tractable when conditioned on $\mathbf{x}_0$:

$$\mathcal{L} = \mathbb{E}_q \left[ \sum\_{t > 1} D\_{\text{KL}}( q(\mathbf{x}\_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \Vert p\_\theta ( \mathbf{x}\_{t-1} | \mathbf{x}_t)) \right]$$

Refer to [the original paper](https://arxiv.org/abs/2006.11239) or our PPT material for more details.

As a parameterization of DDPM, the authors set $\boldsymbol{\Sigma}\_\theta(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}$ to untrained time dependent constants, and they empirically found that predicting noise injected to data by a noise prediction network $\epsilon\_\theta$ is better than learning the mean function $\boldsymbol{\mu}\_\theta$.

In short, the simplified objective function of DDPM is defined as follows:

$$ \mathcal{L}\_{\text{simple}} := \mathbb{E}\_{t,\mathbf{x}_0,\boldsymbol{\epsilon}} [ \Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}\_\theta( \mathbf{x}\_t(\mathbf{x}_0, t), t) \Vert^2  ],$$

where $\mathbf{x}_t (\mathbf{x}_0, t) = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

#### Sampling

Once we train the noise prediction network $\boldsymbol{\epsilon}\_\theta$, we can run sampling by gradually denoising white Gaussian noise. The algorithm of the DDPM  sampling is shown below:

<p align="center">
  <img width="480" alt="image" src="https://github.com/min-hieu/HelloScore/assets/37788686/0722ab08-13cf-4247-8d31-0037875cd71b">
</p>

### TODO

We will generate $64\times64$ animal images using DDPM with AFHQ dataset. We provide skeleton code in wihch you need to fill in missing parts.
You need to construct a noise prediction network according to the provided network diagram and implement the DDPM variance scheduler.
After filling in the missing parts, you can train a model by `python train.py` and generate & save images by

```
python sampling.py --ckpt_path ${CKPT_PATH} --save_dir ${SAVE_DIR_PATH}
```

## Resources
- [[paper](https://arxiv.org/abs/2011.13456)] Score-Based Generative Modeling through Stochastic Differential Equations
- [[paper](https://arxiv.org/abs/2006.09011)] Improved Techniques for Training Score-Based Generative Models
- [[paper](https://arxiv.org/abs/2006.11239)] Denoising Diffusion Probabilistic Models
- [[paper](https://arxiv.org/abs/2105.05233)] Diffusion Models Beat GANs on Image Synthesis
- [[paper](https://arxiv.org/abs/2207.12598)] Classifier-Free Diffusion Guidance
- [[paper](https://arxiv.org/abs/2010.02502)] Denoising Diffusion Implicit Models
- [[paper](https://arxiv.org/abs/2206.00364)] Elucidating the Design Space of Diffusion-Based Generative Models
- [[paper](https://arxiv.org/abs/2106.02808)] A Variational Perspective on Diffusion-Based Generative Models and Score Matching
- [[paper](https://arxiv.org/abs/2305.16261)] Trans-Dimensional Generative Modeling via Jump Diffusion Models
- [[paper](https://openreview.net/pdf?id=nioAdKCEdXB)] Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory
- [[blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)] What is Diffusion Model?
- [[blog](https://yang-song.net/blog/2021/score/)] Generative Modeling by Estimating Gradients of the Data Distribution
- [[lecture](https://youtube.com/playlist?list=PLCf12vHS8ONRpLNVGYBa_UbqWB_SeLsY2)] Charlie's Playlist on Diffusion Processes
- [[slide](./assets/summary_of_DDPM_and_DDIM.pdf)] Juil's presentation slide of DDIM


# Introduction to Score-based Generative Modeling
made by Charlie - hieuristics [at] kaist.ac.kr 
<details>
<summary>Table of Content</summary>

- [Task 0](#task-0-introduction)
- [Task 1](#task-1-very-simple-sgm-pipeline-with-delicious-swiss-roll)
  - [1.1](#1-forward-and-reverse-process) [(a)](#a-ou-process), [(b)](#b-vpsde--vesde)
  - [1.2](#2-training)
  - [1.3](#3-sampling)
  - [1.4](#4-evaluation)
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
├── requirements.txt          <--- required packages 
├── HelloScoreMinimal.ipynb   <--- main code 
├── dataset.py                <--- Define dataset (Swiss-roll, moon, gaussians, etc.)
├── loss.py                   <--- (TODO) Define Training Objective 
├── network.py                <--- (TODO) Define Network Architecture 
├── sampling.py               <--- (TODO) Define Discretization and Sampling 
├── sde.py                    <--- (TODO) Define SDE Processes 
└── train_utils.py            <--- (TODO) Define Training Loop 
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
$$d\mathbf{X}_t = [f(t,\mathbf{X}_t)dt - G(t)^2\nabla_x\log p_t(\mathbf{X}_t)] + G(t)d\bar{\mathbf{B}}_t$$
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
One main different between ISM and DSM is that ISM doesn't require to compute the gradient but instead the divergence. 
In certain cases where it is hard to compute gradient in the [domain of interest](https://arxiv.org/abs/2202.02763) or 
when the problem [naturally contains divergence](https://openreview.net/pdf?id=nioAdKCEdXB), ISM is used.

However, there are other training objectives with their different trade-offs (SSM, EDM, etc.). Highly recommend to checkout 
[A Variational Perspective on Diffusion-based Generative Models and Score Matching](https://arxiv.org/abs/2106.02808) 
and [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) for a more in-depth analysis of the recent training objectives.

**TODO:**
```
- implement your own network in network.py
  (Recommend to implement Positional Encoding, Residual Connection)
- implement ISMLoss in loss.py (hint: you will need to use torch.autograd.grad)
- implement DSMLoss in loss.py
- (optional) implement SSMLoss in loss.py
- implement the training loop in train_utils.py
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

| target distribution | CD | EMD |
|---------------------|----|-----|
| moon                |    |     |
| swiss-roll          |    |     |
| circle              |    |     |

#### 5. [Coming Soon] Schrödinger Bridge (Optional)
One restriction to the typical diffusion processes are that they requires the prior to be easy to sample (gaussian, uniform, etc.). 
Schrödinger Bridge removes this limitation by making the forward process also learnable and allow a diffusion defined between **two** unknown distribution. 
For example, with Schrödinger Bridge, you don't need to know the underlying distribution of the Moon dataset but you can still 
define a diffusion (bridge) that map between the Moon dataset and the swiss roll as shown below. 

Like any diffusion process, there are many ways to learn the Schrödinger Bridge. This section focus on the work presented in 

## Task 2: Image Diffusion [Coming Soon]

## Task 3: Jump Diffusion [Coming Soon] (Optional)


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

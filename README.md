# Introduction to Score-based Generative Modeling
made by Charlie - hieuristics [at] kaist.ac.kr 

## Setup

Install the required package within the `requirements.txt`
```
pip install -r requirements.txt
```

## Code Structure
```
.
├── HelloScoreMinimal.ipynb   <--- Assemble 
├── dataset.py                <--- Define dataset (Swiss-roll, moon, gaussians, etc.)
├── loss.py                   <--- Define Training Objective (TODO)
├── network.py                <--- Define Network Architecture (TODO)
├── requirements.txt          <--- required packages 
├── sampling.py               <--- Define Discretization and Sampling (TODO)
├── sde.py                    <--- Define SDE Processes (TODO)
└── train_utils.py            <--- Define Training Loop (TODO)
```

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

**TODO:**
```
- Derive the expression for the mean and std of the OU process at time t given X0 = 0.
```
## Task 1: Implement a simple pipeline for SGM with delicious swiss-roll
A typical diffusion pipeline is divided into three components:
1. Forward Process and Reverse Process
2. Training
3. Sampling

In this task, we will look into each component one by one and implement them sequentially. 
#### 1. Forward and Reverse Process  
<p align="center">
<img width="364" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/70352400-ac68-4509-b648-f64adcc602bc">
</p>

Our first goal is to setup the forward and reverse processes. In the forward process, the final distribution should be 
the prior distribution which is the standard normal distribution. 
#### (a) OU Process
Following the formulation of the OU Process introduced in the previous section, complete the `TODO` in the 
`sde.py` and check if the final distribution approach unit gaussian.
<p align="center">
<img width="530" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/78fdbb14-b60b-43a9-a90e-a57dd1bcc44a">
</p>

The visualization of the final distribution should look like this:
<p align="center">
<img width="337" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/1ec0aeee-0ac1-4594-85c4-34f56fa47198">
</p>

**TODO:**
```
- implement the forward process using the given marginal probability p_t0(Xt|X0) in SDE.py
- implement the reverse process for general SDE in SDE.py
- (optional) Play around with terminal time (T) and number of time step (N)
  show that the mean and std follow the derived mean and std in task 0
```
#### (b) VPSDE & VESDE
It's mentioned by [Yang Song et al. (2021)](https://arxiv.org/abs/2011.13456) that the DDPM and SMLD are distretization of SDEs. 
Implement this in the `sde.py` and check their mean and and std.

*hint* Although you can simulate the diffusion process through discretization, try getting the explicit equation for the marginal probability $p_{t0}(\mathbf{X}_t \mid \mathbf{X}_0)$

**TODO:**
```
- implement VPSDE in SDE.py
- implement VESDE in SDE.py
- plot the variance of VPSDE and VESDE vs. time
```

#### 2. Training  
The typical training objective of diffusion model uses **D**enoising **S**core **M**atching loss:

$$f_{\theta^*} = \textrm{ argmin }  \mathbb{E} [||f_\theta(\mathbf{X}s) - \nabla p_{s0}(\mathbf{X}_s\mid \mathbf{X}_0)||^2] $$

Where $f_{\theta^*}$ is the optimized score prediction network with parameter $\theta^*$.
However, there are other training objectives with their different trade-offs (ISM, SSM, EDM, etc.). Highly recommend to checkout 
[A Variational Perspective on Diffusion-based Generative Models and Score Matching](https://arxiv.org/abs/2106.02808) 
and [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) for a more in-depth analysis of the recent training objectives.

**TODO:**
```
- implement your own network in network.py
  (Recommend to implement Positional Encoding)
- implement DSM in loss.py
- implement the training loop in train_utils.py
```
#### 3. Sampling  
Finally, we can now use the trained score prediction network to sample from the swiss-roll dataset. Unlike the forward process, there is no analytical form 
of the marginal probabillity. Therefore, we have to run the simulation process. Your final sampling should be close to the target distribution:

<p align="center">
  <img width="161" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/c99737e7-3ebf-4e57-9eb3-5c78b5c334cc">
  <img width="154" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/64661f74-8e64-4d0c-a129-9927f4c57f24">
  <img height="160" alt="image" src="https://github.com/min-hieu/HelloScore/assets/53557912/f32b0483-a115-4748-8bd2-e879e57548d2">
</p>

**TODO:**
```
- implement the SDE discretization.
- (optional) implement the ODE discretization and compare their differences.
```

## Task 2: Implement Image-based Diffusion


## Resources
- [[paper](https://arxiv.org/abs/2011.13456)] Score-Based Generative Modeling through Stochastic Differential Equations
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

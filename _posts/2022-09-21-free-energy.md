---
layout: post
title: "Free energy in physics and statistics"
blurb: ""
img: ""
author: "Andy Jones"
categories: journal
tags: []
<!-- image: -->
---

$$\DeclareMathOperator*{\argmin}{arg\,min}$$
$$\DeclareMathOperator*{\argmax}{arg\,max}$$

<style>
.column {
  float: left;
  width: 30%;
  padding: 5px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

## Free energy in physics

## Free energy in statistics

Consider a generic generative model:

$$x \sim p(x | \theta),~~\theta \sim p(\theta).$$

Suppose we'd like to compute the posterior $p(\theta | x)$ but it's intractable, so we use a variational approximating family $q_\phi(\theta),$ where $\phi$ is the variational parameter. A common way to fit the variational approximation is to maximize a lower bound on the model evidence. Note that we can decompose the log evidence as follows:

\begin{align}
\log p(x) &= \log \frac{p(x | \theta) p(\theta)}{p(\theta | x)} \\\
&= \mathbb{E}\_q\left[ \log \left\{\frac{q(\theta)}{q(\theta)} \frac{p(x | \theta) \log p(\theta)}{p(\theta | x)} \right\} \right] \\\
&= \color{red}{\mathbb{E}\_q\left[ \log \frac{q(\theta)}{p(\theta | x)}\right]} + \color{blue}{\mathbb{E}\_q\left[\log \frac{p(x | \theta) p(\theta}{q(\theta)} \right]} \\\
&= \color{red}{D_{KL}(q \|\| p)} + \color{blue}{\mathcal{F}}.
\end{align}

Here $\mathcal{F}$ is the free energy. It's also often called the evidence lower bound (ELBO) because $\log p(x) \geq \mathcal{F}$ since the KL divergence is always non-negative. 

We can see that the log evidence $\log p(x)$ decomposes into the sum of two terms: a KL divergence and the free energy. In some sense, we can think of the KL divergence as the "distance" between the posterior $p(\theta | x)$ and the appoximating distribution $q(\theta)$. The KL divergence is analogous to the amount of *lost, unrecoverable energy*. The free energy $\mathcal{F}$ is the energy that we're allowed to play with, and usually we want as much free energy as possible. In other words, we want to lose as little energy as possible to the KL divergence term.

We can rearrange terms to write the free energy as follows:

$$\mathcal{F} = \log p(x) - D_{KL}(q \|\| p).$$








---
layout: post
title: "Monge and Kontorovich formulations of the Optimal Transport problem"
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

_Transportation theory_ studies mathematical and algorithmic formulations of the physical movement of goods and resource allocation. The field of _optimal transport_ is concerned with finding routes for these movements that minimize some type of cost.

## Preliminaries

We denote a discrete density over a support at $n$ different locations $x_1, \dots, x_n$ as

$$\alpha = \sum\limits_{i=1}^n a_i \delta_{x_i}$$

where $\delta_{x_i}$ is a Dirac point mass at location $x_i$ and $\mathbf{a} = (a_1, \cdots, a_n)^\top$ is a vector of weights for each position. We furthermore constrain the weights to sum to one, $\sum_{i=1}^n a_i = 1$, and thus $\mathbf{a}$ lies on a $n$-dimensional simplex, which we denote as $\mathbf{a} \in \Delta^n$.

## Monge formulation

The Monge formulation restricts itself to transportation between two uniform discrete densities whose support has the same size $n$. In particular, it seeks a mapping from one density $\alpha$ to another $\beta$, where

$$\alpha = \sum\limits_{i=1}^n a_i \delta_{x_i},~~~\beta = \sum\limits_{j=1}^n b_j \delta_{y_j}$$

and $a_1 = \cdots = a_n = b_1 = \cdots b_n = \frac1n$. Note that the supports of the two densities, $\\{x_i\\}\_{i=1}^n$ and $\\{y_j\\}\_{j=1}^n$, may not be the same, and in fact need not be overlapping at all.

Furthermore, we assume that there is a cost associated with assigning a mass at $x_i$ to be transported to $y_i$. In this case, we can represent this cost function as a matrix $\mathbf{C} \in \mathbb{R}\_+^{n \times n}$. The $ij$'th element, $\mathbf{C}\_{ij}$ will be a nonnegative scalar representing the cost of transporting from $x_i$ to $y_j$.

In this simple case, the optimal transport problem reduces to a problem of "matching," where we'd like to pair each $x_i$ with a single $y_j$ such that the cost is minimizes. Formally, we can identify this as a problem of finding a permutation $\sigma : [n] \rightarrow [n]$ that maps the indices $1, \dots, n$ to the indices $1, \dots, n$.

$$\min_{\sigma \in \text{Perm}(n)} \sum\limits_{i=1}^n \mathbf{C}_{i, \sigma(i)}.$$

## Kontorovich formulation

The Kontorovich version of optimal transport is a strict generalization of the Monge version. Rather than assuming that each mass is transported to just one other location, the Kontorovich problem allows for it to be split among multiple locations. Furthermore, it no longer requires each density to be uniform over the same size support.

To formalize this problem, consider again two discrete densities,

$$\alpha = \sum\limits_{i=1}^n a_i \delta_{x_i},~~~\beta = \sum\limits_{j=1}^m b_j \delta_{y_j}$$

where $n \neq m$ in general, and $\mathbf{a}$ and $\mathbf{b}$ are any two $n$- and $m$-dimensional (respectively) positive vectors.

The goal is now to find a matrix $\mathbf{P} \in \mathbb{R}\_+^{n \times m}$ whose $ij$'th element is a scalar representing the amount of mass at location $x_i$ that shuold be transported to location $y_j$. 

Importantly, we have extra constraints on this matrix: We must ensure that mass is conserved in the transportation plan. In other words, the total amount of mass distributed from $x_i$ to all target locations must equal $a_i$, and the amount of mass distributed _to_ $y_j$ from all source locations must be equal to $b_j$. Stating this mathematically, we can write the set of admissible transport plans as 

$$\mathbf{U}(\mathbf{a}, \mathbf{b}) = \left\{\mathbf{P} \in \mathbb{R}_+^{n \times m} : \mathbf{P} \mathbf{1}_m = \mathbf{a}, \mathbf{P}^\top \mathbf{1}_n = \mathbf{b} \right\}.$$

### Extension to continuous distributions

All of the above descriptions have focused on discrete densities. However, we can also consider the problem of transporting one continuous density to another continuous density. (Semi-discrete OT problems also exist, where a continuous density is transported to a discrete density.)

This case is described most easily in terms of marginal and joint distributions. We observe two marginal distributions $p(x)$ and $p(y)$, and we're interested in finding a transport plan that maps $p(x)$ to $p(y)$. In the continuous case, this plan can be described with a joint density over both $x$ and $y$.

$$\left\{p(x, y) : \int p(x, y) dy = p(x), \int p(x, y) dx = p(y)\right\}$$

## Examples

### Continuous densities

As a simple starting example, consider two densities,

$$x \sim \mathcal{N}(0, 1),~~~y \sim \mathcal{N}(0, 1).$$

These marginals are plotted below, adn we'd like to find the joint $p(x, y)$ that minimizes a cost.

<center>
<figure>
  <img src="/assets/ot_marginal_densities.png">
  <figcaption><i></i></figcaption>
</figure>
</center>

Note that there are infinitely many joint distributions that are consistent with having observed these marginals. Below we've plotted the contours three such two-dimensional densities.

<div class="row">
  <div class="column">
    <img src="/assets/ot_joint_density1.png" alt="joint1" style="width:100%">
  </div>
  <div class="column">
    <img src="/assets/ot_joint_density2.png" alt="joint2" style="width:100%">
  </div>
  <div class="column">
    <img src="/assets/ot_joint_density3.png" alt="joint3" style="width:100%">
  </div>
</div>

Let the cost be the squared Euclidean distance:

$$c(x, y) = \|x - y\|^2.$$

Then our optimization problem is

\begin{align}
\min_{p(x, y)} \int c(x, y) p(x, y) dx dy &= \min_{p(x, y)}\mathbb{E}\_{p(x, y)}\left[ c(x, y) \right] \\\
&= \min_{p(x, y)}\mathbb{E}\_{p(x, y)}\left[ \\|x - y\\|^2 \right].
\end{align}






---
layout: post
title: "Natural gradients"
blurb: ""
img: ""
author: "Andy Jones"
categories: journal
tags: []
<!-- image: -->
---

$$\DeclareMathOperator*{\argmin}{arg\,min}$$

## Gradient descent

Recall that gradient descent is an algorithm that iteratively takes a step in the direction of the negative gradient,

$$x_{t+1} = x_t - \gamma \nabla f(x_t)$$

where $\gamma$ is a learning rate parameter, and $\nabla f(x_t)$ is the gradient of the objective function at $x_t$,

$$\nabla f(x_t) = 
\begin{bmatrix} 
\frac{\partial}{\partial x_t^{(1)}} f(x_t) \\ 
\frac{\partial}{\partial x_t^{(2)}} f(x_t) \\ 
\vdots \\ 
\frac{\partial}{\partial x_t^{(p)}} f(x_t)
\end{bmatrix}.$$

## Steepest descent

Gradient descent is a special case of an algorithm/design principle known as "steepest descent." The principle of steepest descent basically says that out of all the steps we could take that have a given fixed length, we should choose the one that reduces the cost function the most. 

How we measure the length of these steps depends on the geometry of the problem. Gradient descent is the result of applying the steepest descent idea to the realm of Euclidean geometry.

To show this, consider a linear approximation of $f$ around $x_t$,

$$f(x) \approx f(x_t) + \nabla f(x_t)^\top (x - x_t).$$

We'd like to choose an $x_{t+1}$ to minimize this. However, we need to add more constraints, since in this unconstrained problem the optimal move would be to take an infinite step in the direction of the negative gradient.

Suppose that, in addition to minimizing $f$, we simulatneously minimize one-half of the squared Euclidean distance between $x_t$ and $x_{t+1}$. This penalizes large steps. Using the linear approximation of $f$, the optimization problem then becomes

$$x_{t+1} = \argmin_x f(x_t) + \nabla f(x_t)^\top (x - x_t) + \frac12 \|x - x_t\|^2_2.$$

Taking the gradient with respect to $x$ and setting it to zero, we have

$$\nabla f(x_t) + x_{t+1} - x_t = 0.$$

Finally, we have

$$x_{t+1} = x_t - \nabla f(x_t),$$

which corresponds to a typical gradient descent update. Notice that we can impose a learning rate $\gamma$ by penalizing a scaled Euclidean distance instead,

$$\frac{\gamma}{2} \|x - x_t\|^2_2.$$

## Generalizing the metric

However, this choice of a Euclidean metric is a very special case, and we can generalize it. In particular, consider an arbitrary metric $D(x, x^\prime)$ that measures a dissimilarity or distance. Our general optimization problem is then

$$x_{t+1} = \argmin_x f(x_t) + \nabla f(x_t)^\top (x - x_t) + D(x, x_t).$$

[Amari](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf) showed that the steepest descent direction in general depends on the Riemannian metric tensor of the parameter space. Recall that the Pythagorean theorem says that for a $p$-dimensional parameter space and two vectors $x_t$ and $x_{t+1}$, the squared length of the vector between them $dx$ is

$$|dx|^2 = \sum\limits_{i=1}^p (dx_i)^2$$

where $dx_i$ is the $i$th element of $dx$. We can obtain a generalized version of the Pythagorean theorem for a Riemannian manifold:

$$|dx|^2 = \sum\limits_{i, j \in [p]} g_{ij}(x) dx_i dx_j.$$

If we group the scaling terms $g_{ij}$ into one $p\times p$ matrix $G$, this matrix is known as the Riemannian metric tensor. Amari shows that the steepest descent direction in a space with metric tensor $G$ is given by

$$-G^{-1}(x_t) \nabla f(x_t).$$

We can note that this again reduces to gradient descent when $G$ is the identity matrix.

### Mahalanobis metric

A slight generalization of the Euclidean distance is the Mahalanobis metric,

$$D(x, x^\prime) = (x - x^\prime)^\top A (x - x^\prime)$$

where $A$ is a PSD $p \times p$ matrix.

To start, consider the case where $A = \text{diag}(a_1, \cdots, a_p)$ is a diagonal matrix with positive diagonal elements. Notice that the resulting natural gradient will result in a simple element-wise rescaling of the gradient,

$$x_{t+1} = x_t - A^{-1} \nabla f(x_t) = x_t - \rho^\top \nabla f(x_t)$$

where $\rho = [1/a_1,\cdots, 1/a_p]^\top$.

This reduces to the Euclidean metric with $A$ is the identity matrix -- in other words $a_1 = a_2 = \cdots = a_p = 1$.

With a non-diagonal $A$, we can account for different warping of our parameter space. Heuristically, we can think of $A$ as accounting for rotations and stretching away from the Euclidean metric. Under the Euclidean metric, equidistant points form a sphere; under the Mahalanobis metric, they form an ellipsoid.

<center>
<figure>
  <img src="/assets/equidistant_points.png">
  <figcaption>Equidistant points under different metrics (distance measured from the origin here).</figcaption>
</figure>
</center>

The update in this case is equivalent to Newton's method because the structure of the loss function is quadratic:

$$x_{t+1} = x_t - A^{-1} \nabla f(x_t) = x_t - A^{-1} \nabla f(x_t).$$

As a simple example, consider the loss function

$$f(x) = x^\top A x,~~~~A = 
\begin{bmatrix}
1 & 1/2 \\
1/2 & 1
\end{bmatrix}.$$

In the left figure below, we apply vanilla gradient descent under this loss function. We can see that the steps bounce around the "valley" formed by $f$.

In the right figure, we perform steepest descent using the metric $(x - x^\prime)^\top A (x - x^\prime).$ We can see that the steps go straight toward the global minimum in this case.

<center>
<figure>
  <img src="/assets/gd_euclidean_mahalanobis.png">
  <figcaption>Steepest descent under Euclidean and Mahalanobis metrics.</figcaption>
</figure>
</center>

However, in practice we won't know the full geometry of the loss function. In the toy examples above, we only knew $A$ because we constructed the problem ourselves. Thus, another challenge is locally estimating the curvature of the space.

### Adam, etc.

### Fisher metric

When working with probabilistic models, we can use yet another metric that's more friendly to probably distributions: the KL divergence. Suppose we have a joint probability model $p(\mathcal{D}, \theta)$, where $\mathcal{D}$ is some data, and $\theta$ is a parameter vector. Suppose $\theta_t$ is a vector of the current parameter values, and we'd like to find a new set of parameters $\theta_{t+1}$. To do so under the natural gradient framework, we solve the following problem.

$$\theta_{t+1} = \argmin_\theta f(\theta_t) + \nabla f(\theta_t)^\top (\theta - \theta_t) + D_{KL}(p(x, \theta) \| p(x, \theta_t)).$$

Unfortunately this minimization is intractable in general. However, we can approximate the KL divergence using a second-order Taylor expansion, which turns out to be the Fisher information matrix $F$ (see [Appendix](#appendix) for derivation). This means that locally around $\theta_t$, we have

$$D_{KL}(p(x | \theta) \| p(x | \theta_t) \approx F.$$

where

$$F = \mathbb{E}\left[(\nabla_\theta \log p(\mathcal{D}, \theta))(\nabla_\theta \log p(\mathcal{D}, \theta))^\top\right]$$

### Gaussian example

Let $x \sim \mathcal{N}(\mu, \sigma^2)$, and suppose we parameterize the Gaussian in terms of its mean $\mu$ and log standard deviation $\lambda = \log \sigma$, with $\theta = (\mu, \lambda)$. The log density is

$$\log p(x | \theta) = -\frac12 \log 2\pi - \lambda - \frac12 \exp(-2\lambda) (x - \mu)^2.$$

Taking the gradient, we have

\begin{align}
\nabla_\theta \log p(x | \theta) &= 
\begin{bmatrix}
\exp(-2\lambda) (x - \mu) \\\
-1 + \exp(-2\lambda)(x - \mu)^2
\end{bmatrix} \\\
&= \begin{bmatrix}
\frac{1}{\sigma^2} (x - \mu) \\\
-1 + \frac{1}{\sigma^2} (x - \mu)^2
\end{bmatrix}.
\end{align}

We can then compute the Fisher information matrix. Recall that the Fisher is the expectation of the outer product of the gradient:

$$F = \mathbb{E}\left[[\nabla_\theta \log p(x | \theta)] [\nabla_\theta \log p(x | \theta)]^\top\right].$$

Plugging in the gradient for the Gaussian, we have

$$F = \begin{bmatrix}
\frac{1}{\sigma^2} (x - \mu) \\\
-1 + \frac{1}{\sigma^2} (x - \mu)^2
\end{bmatrix} 
\begin{bmatrix}
\frac{1}{\sigma^2} (x - \mu) \\\
-1 + \frac{1}{\sigma^2} (x - \mu)^2
\end{bmatrix}^\top.$$

Expanding the outer product, we have

$$F = \begin{bmatrix}
\mathbb{E}\left[\frac{1}{\sigma^4} (x - \mu)^2\right] & \mathbb{E}\left[-\frac{1}{\sigma^2} (x - \mu) + \frac{1}{\sigma^4} (x - \mu)^3\right] \\\
\mathbb{E}\left[-\frac{1}{\sigma^2} (x - \mu) + \frac{1}{\sigma^4} (x - \mu)^3\right] & \mathbb{E}\left[1 - \frac{2}{\sigma^2} (x - \mu)^2 + \frac{1}{\sigma^4} (x - \mu)^4\right]
\end{bmatrix}.$$

The off-diagonal terms will be $0$ because $\mathbb{E}[x - \mu] = \mathbb{E}[(x - \mu)^3] = 0.$ Using the definition of the variance, the top-left term will be

$$\frac{1}{\sigma^4}\mathbb{E}[(x - \mu)^2] = \frac{\sigma^2}{\sigma^4} = \frac{1}{\sigma^2}.$$

Finally, using the fact that $\mathbb{E}[(x - \mu)^3] = 3\sigma^4$, we can find the lower-right term:

$$1 - \mathbb{E}\left[\frac{2}{\sigma^2} (x - \mu)^2 + \frac{1}{\sigma^4} (x - \mu)^4\right] = 1 - \frac{2\sigma^2}{\sigma^2} + \frac{3\sigma^4}{\sigma^4} = 2.$$

Bringing it all together, we see that the Fisher information matrix is

$$F = \begin{bmatrix}
\frac{1}{\sigma^2} & 0 \\\
0 & 2
\end{bmatrix}.$$

Clearly, the inverse Fisher is given by

$$F = \begin{bmatrix}
\sigma^2 & 0 \\\
0 & \frac12
\end{bmatrix}.$$

Looking at the isocontours of the univariate Gaussian, we can see that it's useful to scale the mean's gradient by the variance because the geometry becomes much steeper as the variance gets lower.

<center>
<figure>
  <img src="/assets/gaussian_isocontours_fisher.png">
  <figcaption>Steepest descent under Euclidean and Mahalanobis metrics.</figcaption>
</figure>
</center>

## Code

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse
from os.path import join as pjoin
inv = np.linalg.inv
import matplotlib
font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True


A = np.array(
        [[1, -0.5],
        [-0.5, 1]]
)

f = lambda x1, x2: x1**2 + x2**2 - 1 * x1 * x2
fgrad = lambda x1, x2: np.array([2 * x1 - x2, 2*x2 - x1])


plt.figure(figsize=(14, 7))

plt.subplot(121)
xlimits = [-10, 10]
ylimits = [-10, 10]
numticks = 100
x = np.linspace(*xlimits, num=numticks)
y = np.linspace(*ylimits, num=numticks)
X, Y = np.meshgrid(x, y)
zs = f(np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel()))
Z = zs.reshape(X.shape)
plt.contour(X, Y, Z, levels=30)

xhat = np.array([8., 3.])

for _ in range(10):
        g = -0.5 * fgrad(*xhat)
        plt.arrow(*xhat, *g, head_width=0.5, color="black")
        xhat += g

plt.title(r"$x_{t+1} = x_t - \gamma \nabla f(x)$")
plt.xticks([])
plt.yticks([])


plt.subplot(122)
xlimits = [-10, 10]
ylimits = [-10, 10]
numticks = 100
x = np.linspace(*xlimits, num=numticks)
y = np.linspace(*ylimits, num=numticks)
X, Y = np.meshgrid(x, y)
zs = f(np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel()))
Z = zs.reshape(X.shape)
plt.contour(X, Y, Z, levels=30)

xhat = np.array([8., 3.])

for _ in range(10):
        g = -0.3 * inv(A) @ fgrad(*xhat)
        plt.arrow(*xhat, *g, head_width=0.5, color="black")
        xhat += g

plt.title(r"$x_{t+1} = x_t - \gamma A^{-1} \nabla f(x)$")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

import ipdb; ipdb.set_trace()

```

## References
- Amari, Shun-Ichi. "Natural gradient works efficiently in learning." Neural computation 10.2 (1998): 251-276.
- Martens, James. "New insights and perspectives on the natural gradient method." arXiv preprint arXiv:1412.1193 (2014).
- Ollivier, Yann, et al. "Information-geometric optimization algorithms: A unifying picture via invariance principles." Journal of Machine Learning Research 18.18 (2017): 1-65.
- Prof. Roger Grosse's [course notes](https://csc2541-f17.github.io/slides/lec05a.pdf).

## Appendix

### Fisher information approximates the KL divergence

For notational simplicity, let $D(\theta, \theta_t) = D_{KL}(p_\theta(x) \| p_{\theta_t}(x))$. Consider a second-order Taylor approximation to the KL divergence around $\theta_t$,

$$D(\theta, \theta_t) \approx D(\theta_t, \theta_t) + \left(\nabla_\theta D(\theta, \theta_t)\big|_{\theta = \theta_t}\right)^\top (\theta - \theta_t) + (\theta - \theta_t)^\top H_t(\theta - \theta_t)$$

where $H_t$ is the Hessian of $D(\theta_t, \theta_t)$ at $\theta_t$.

The first two terms are zero. The first term is a divergence between two equal distributions, which makes the divergence zero. For the second term, we can see that

\begin{align}
	\nabla_\theta D(\theta, \theta_t) &= \nabla_\theta \mathbb{E}\_{p(x | \theta)}\left[\log \frac{p(x | \theta)}{p(x | \theta_t)}\right] \\\
	&= \mathbb{E}\_{p(x | \theta)}\left[\nabla_\theta \log \frac{p(x | \theta)}{p(x | \theta_t)}\right] & \text{(Swap $\nabla$ and $\mathbb{E}$)} \\\
	&= \mathbb{E}\_{p(x | \theta)}\left[\nabla_\theta \log p(x | \theta)\right] & \text{(Grad. doesn't depend on $\theta_t$)} \\\
	&= 0.
\end{align}

The final line comes from the fact that the expectation of the score is always $0$.




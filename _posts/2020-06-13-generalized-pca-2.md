---
layout: post
title: "Generalized PCA: an alternative approach"
author: "Andy Jones"
categories: journal
blurb: ""
img: ""
tags: []
<!-- image: -->
---


Principal component analysis is a widely-used dimensionality reduction technique. However, PCA has an implicit connection to the Gaussian distribution, which may be undesirable for non-Gaussian data. Here, we'll see a second approach for generalizing PCA to other distributions introduced by [Andrew Landgraf in 2015](https://etd.ohiolink.edu/!etd.send_file?accession=osu1437610558&disposition=inline).

## Introduction

In its traditional form, PCA makes very few assumptions. Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ be our data matrix. Then, to find the first component, we seek a vector $\mathbf{u} \in \mathbb{R}^p$ such that the variance of the linear combination $\mathbf{X}\mathbf{u}$ is maximized. In particular, we solve

$$\mathbf{u} = \text{arg}\max_{||u|| = 1} \text{var}(\mathbf{X}\mathbf{u}) = \text{arg}\max_{||u|| = 1} \mathbf{u}^\top \mathbf{S}_n \mathbf{u}$$

where $\textbf{S}_n$ is the sample covariance matrix of $\mathbf{X}$. To find subsequent PCs, we would solve a similar optimization problem for $\mathbf{u}_2, \dots, \mathbf{u}_p$, with the additional constraint that the subsequent PCs are orthogonal to the preceding ones.

There are several other interpretations and solutions for PCA, such as casting it as an eigendecomposition of $\mathbf{X}^\top \mathbf{X}$, or as a minimization of the Frobenius norm of the data projected onto an orthogonal basis:

$$\hat{\mathbf{U}} = \text{arg}\max_\mathbf{U} ||XB||$$

with the constraint $\mathbf{U}^\top \mathbf{U} = \mathbf{I}$.

In the above formulations, the primary assumption is linearity.

Consider one last interpretation of PCA: as the maximum likelihood solution to a probabilistic model. This was [Tipping and Bishop's 1999 approach](https://www.apps.stat.vt.edu/leman/VTCourses/PPCA.pdf) when they formulated probabilistic PCA. 

In particular, probabilistic PCA assumes that there exist some latent, lower-dimensional variables $\mathbf{z}_1, \dots, \mathbf{z}_n \in \mathbb{R}^k$ where $k < p$, such that the data $\mathbf{x}_1, \dots, \mathbf{x}_n$ can be faithfully represented in this latent variables. Under a Gaussian model we would assume that a data vector $\mathbf{x} \in \mathbb{R}^p$ has the distribution

$$\mathbf{x}_i | \mathbf{z}_i \sim \mathcal{N}(\mathbf{U} \mathbf{z}_i, \sigma^2 \mathbf{I})$$

where $\mathbf{U} \in \mathbb{R}^{p \times k}$ in this case, and $\mathbf{z}_i \sim \mathcal{N}(0, \mathbf{I})$.

[It can be shown](https://etd.ohiolink.edu/!etd.send_file?accession=osu1437610558&disposition=inline) that the maximum likelihood solution to this probabilistic model and the solution for traditional PCA are nearly identical. Thus, traditional PCA is strongly related to an assumption of the data being Gaussian distributed.

However, if the data follow a non-Gaussian distribution, this may be undesirable. There have been a few proposals for generalizations of PCA to non-Gaussian distributions, one of which we saw in an [earlier post](https://andrewcharlesjones.github.io/posts/2020/03/generalizedpca/). Here, we'll see a second approach to generalizing PCA to the exponential family.

## Generalized PCA

Recall the general form of the exponential family of distributions:

$$f(x) = \exp\left(\frac{x \theta}{a(\phi)} + c(x, \phi)\right)$$

where $\theta$ is the canoical natural parameter, and $\phi$ is a dispersion parameter.

In the earlier approach, we saw how [Collins et al.](https://papers.nips.cc/paper/2078-a-generalization-of-principal-components-analysis-to-the-exponential-family.pdf) used the theory of generalized linear models to easily substitute in any exponential family likelihood in order to generalize PCA. In particular, they factorized the parameter matrix $\mathbf{\Theta}$ as $\mathbf{\Theta} = \mathbf{A} \mathbf{B}^\top$, and then maximized the negative log-likelihood, where any exponential family likelihood can be substituted in.

## Generalized PCA as a projection

As an alternative generalization of PCA, [Landgraf](https://etd.ohiolink.edu/!etd.send_file?accession=osu1437610558&disposition=inline) framed PCA as a projection of the natural parameters to a lower-dimensional space.

As we saw above, PCA can be seen as finding the matrix $\mathbf{U}$ (where $\mathbf{U}^\top \mathbf{U} = \mathbf{I}$) that minimizes the following:

$$||\mathbf{X} - \mathbf{X} \mathbf{U} \mathbf{U}^\top||_F^2.$$

In terms of a GLM, the above formulation is equivalent to minimizing the deviance of a Gaussian model with known variance. Recall that the deviance essentially measures the log-likelihood difference between the "saturated" (full) model and the fitted model.

In the Gaussian case, the natural parameter is equal to the data: $\mathbf{\Theta} = \mathbf{X}$, and the link function is the identity. Thus, the deviance has the form of a sum of squares:

$$D = \frac{1}{\sigma^2} \sum\limits_{i=1}^n (\mathbf{x}_i - \mathbf{x}_i \hat{\theta}_i)^2$$

where $\hat{\theta}_i$ are the estimated natural parameters.

Now, we'll consider estimating the natural parameter matrix as

$$\Theta = \widetilde{\Theta} \mathbf{U} \mathbf{U}^\top$$

where $\widetilde{\Theta}$ are the natural parameters of the saturated model.

Importantly, notice that in this approach, we aren't completely decomposing $\Theta$ into two submatrices, but rather projecting it onto an orthogonal basis.

We can then formulate the objective function as minimizing the deviance between the PCA model and the saturated model:

\begin{align} D(\mathbf{X}; \widetilde{\Theta} \mathbf{U} \mathbf{U}^\top) &= -2 \underbrace{\log f(\mathbf{X}; \widetilde{\Theta} \mathbf{U} \mathbf{U}^\top)}\_{\text{$LL$ for PCA model}} \;\; + \underbrace{2\log f(X; \widetilde{\Theta})}\_{\text{$LL$ for saturated model}} \\\ &\propto -\langle \mathbf{X}, \widetilde{\Theta} \mathbf{U} \mathbf{U}^\top \rangle + \sum\limits_{i=1}^n \sum\limits_{j=1}^p b\left([\mathbf{U} \mathbf{U}^\top \widetilde{\theta}\_i]\_j \right) \\\ \end{align}

where $b(\cdot)$ depends on the chosen exponential family model.

## Advantages of this approach

The primary advantage of Landgraf's approach is that the formulation only needs to solve for the PC loadings $\mathbf{U}$ without worrying at all about the PC scores, such as is done in the Collins approach. In other words, instead of decomposing the natural parameters as

$$\Theta = \mathbf{A}\mathbf{B}^\top$$

we decompose them as 

$$\Theta = \widetilde{\Theta}\mathbf{U}\mathbf{U}^\top.$$

This implies another advantage: if we have some held-out data $\mathbf{x}^*$, we can calculate the PC scores with simple matrix computation:

$$\hat{U}^\top \mathbf{\widetilde{\theta}}^*$$

where $\mathbf{\widetilde{\theta}}^*$ are the natural parameters for $\mathbf{x}^*$ under the saturated model. In contrast, under the Collins approach, solving for the PC scores on held-out data would require re-running an entire optimization problem.


## References

- Hotelling, Harold. "Analysis of a complex of statistical variables into principal components." Journal of educational psychology 24.6 (1933): 417.
- Tipping, Michael E., and Christopher M. Bishop. "Probabilistic principal component analysis." Journal of the Royal Statistical Society: Series B (Statistical Methodology) 61.3 (1999): 611-622.
- Prof. Jonathan Pillow's [notes on PCA](http://pillowlab.princeton.edu/teaching/statneuro2018/slides/notes05_PCA2.pdf)
- Landgraf, Andrew J. Generalized principal component analysis: dimensionality reduction through the projection of natural parameters. Diss. The Ohio State University, 2015.

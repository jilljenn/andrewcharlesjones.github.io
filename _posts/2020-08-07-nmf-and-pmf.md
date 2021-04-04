---
layout: post
title: "Connection between non-negative matrix factorization and Poisson matrix factorization"
author: "Andy Jones"
categories: journal
tags: []
<!-- image: -->
---


In this post, we draw a simple connection between the optimization problems for NMF and PMF.


## NMF


Consider a data matrix $X \in \mathbb{R}^{n \times p}$ where every value is zero or a positive integer, $x_{ij} \in \mathbb{N}_0$. NMF tries to find a low-rank approximation of $X$ such that 

$$X \approx WH^\top,$$

where $W$ is $n \times k$, $H$ is $p \times k$, and $k < \min(n, p)$. The entries of $W$ and $H$ are constrained to be nonnegative as well. 

NMF can use a variety of objective functions to optimize $W$ and $H$. A commonly used loss function is a form of the the KL-divergence between $X$ and $WH^\top$. Recall that the KL-divergence between to distributions $p$ and $q$ is defined as 

$$\text{KL}(p || q) = \sum\limits_x p(x) \log \frac{p(x)}{q(x)}.$$

In the case of NMF, if $X$ and $WH^\top$ are valid probability distributions --- such that $\sum\limits_{i, j} x_{ij} = 1$ and $\sum\limits_{ij} (WH^\top)\_{ij} = 1$ --- then we can write the KL-divergence between $X$ and $WH^\top$ as

$$\text{KL}(X || WH^\top) = \sum\limits_{i, j} x_{ij} \log \frac{x_{ij}}{(WH^\top)_{ij}}.$$

NMF often uses a form of the KL-divergence that accounts for cases when $X$ and $WH^\top$ don't properly sum to $1$. It achieves this by optimizing the following objective function, often generically called the "divergence" beteween $X$ and $WH^\top$:

$$\mathcal{L}_{\text{NMF}} = \sum\limits_{i=1}^n \sum\limits_{j=1}^p \left( x_{ij} \log \frac{x_{ij}}{(WH^\top)_{ij}} - x_{ij} + (WH^\top)_{ij}\right).$$

This divergence adds a linear penalty so that the overall sums of $X$ and $WH^\top$ are similar.

## Poisson matrix factorization

Consider again a matrix of counts $X$, where $x_{ij} \in \mathbb{N}\_0$, and the $x_{ij}$'s are independent. If $x_{ij}$ is modeled to have a Poisson distribution with mean $\mu_{ij}$, then the likelihood is

$$\mathcal{L}_{\text{Po}} = \prod\limits_{i=1}^n \frac{\mu_{ij}^{x_{ij}} e^{-\mu_{ij}}}{x_{ij}!}.$$

Taking the logarithm and ignoring the $x_{ij}!$ terms since they're constant w.r.t. the parameters, we have

$$\log \mathcal{L}_{\text{Po}} = \sum\limits_{i=1}^n x_{ij} \log \mu_{ij} - \mu_{ij}.$$

If we take a maximum likelihood approach, maximizing the log-likelihood will yield the equivalent solution to minimizing the negative log-likelihood, $\sum\limits_{i=1}^n -x_{ij} \log \mu_{ij} + \mu_{ij}$. We can then add a constant term that only depends on the data, $x_{ij} \log x_{ij} - x_{ij}$:

\begin{align} \text{arg} \min_{\mu_{ij}} (-\log \mathcal{L}\_{\text{Po}}) &= \text{arg} \min_{\mu_{ij}} \sum\limits_{i=1}^n \left(-x_{ij} \log \mu_{ij} + \mu_{ij} + x_{ij} \log x_{ij} - x_{ij}\right) \\\ &= \text{arg} \min_{\mu_{ij}} \sum\limits_{i=1}^n \left( x_{ij} \log \frac{x_{ij}}{\mu_{ij}} + \mu_{ij} - x_{ij} \right). \\\ \end{align}

This last expression has the same form as the divergence in NMF.

Consider the matrix of Poisson parameters $\boldsymbol{\mu} \in \mathbb{R}^{n \times p}$. If we were to factorize this matrix into two smaller matrices such that $\boldsymbol{\mu} \approx WH^\top$, then we would exactly recover the NMF objective function.

## Conclusion

By showing the connection between the Poisson likelihood and the KL-divergence, we've drawn a connection between NMF and the maximum likelihood estimator for Poisson matrix factorization.

## References

- Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." Advances in neural information processing systems. 2001.
- Gopalan, Prem, Jake M. Hofman, and David M. Blei. "Scalable recommendation with poisson factorization." arXiv preprint arXiv:1311.1704 (2013).
- Devarajan, Karthik, Guoli Wang, and Nader Ebrahimi. "A unified statistical approach to non-negative matrix factorization and probabilistic latent semantic indexing." Machine learning 99.1 (2015): 137-163.

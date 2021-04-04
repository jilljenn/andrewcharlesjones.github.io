---
layout: post
title: "Reduced-rank regresssion"
author: "Andy Jones"
categories: journal
tags: []
<!-- image: -->
---


Reduced-rank regression is a method for finding associations between two high-dimensional datasets with paired samples.

## Setup

Consider two datasets, $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$, that have measurements on the same $n$ samples, but different features (each sample in $X$ has dimensionality $p$, and each sample in $Y$ has dimensionality $q$).

Suppose we'd like to understand the relationship between these two data matrices by running some sort of regression. The simplest way to do this would be to run several instances of multiple regression. For example, if we deem $X$ to be the "covariates", and $Y$ to be the response, we could run $q$ separate multiple regression models $M_1, \dots, M_q$, each of which has the same $p$ covariates, but where model $M_j$ only has the $j$th covariate of $Y$ as the response.

This approach would generate a $p \times q$ table of coefficients, where each column is the association of $X$ with the corresponding feature of $Y$, independent of all other features of $Y$.

However, in many cases it's useful to jointly consider all of the covariates in $Y$, rather than treating them separately. There is a fairly large family of models that try to solve this problem, such as canonical correlation analysis, partial least squares, and reduced-rank regression. Below, we'll review reduced-rank regression and see what sets it apart from other, similar models.

## Simple, multiple, and multiple multivariate regression

Recall that a univariate regression attempts to approximate a response vector $Y \in \mathbb{R}^n$ with a linear map from $X \in \mathbb{R}^n$ via a scalar coefficient $\beta$ and some noise $\epsilon$:

$$Y = X \beta + \epsilon.$$

The ordinary least squares solution to this problem is $\hat{\beta} = (X^\top X)^{-1} X^\top Y$.

In *multiple* linear regression, the design matrix $X$ has more than one feature, $X \in \mathbb{R}^{n \times p}$, and we thus seek a coefficient vector $\beta \in \mathbb{R}^{p}$ to approximate $Y \in \mathbb{R}^n$ -- that is, again $Y \approx X\beta$.

In *multiple multivariate* regression, both $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$ are multidimensional, and we seek a coefficient matrix $B \in \mathbb{R}^{p \times q}$ to approximate the linear map between $X$ and $Y$: again $Y = XB + \epsilon$.

## Reduced-rank regression

Reduced-rank regression (RRR) is a variant of multiple multivariate regression, with an added constraint: rather than estimating $\beta$ as a $p \times q$ matrix of coefficients, RRR enforces that $\text{rank}(\beta) = r$, where $r < \min(p, q)$. Intuitively, this constraint enforces the assumption that $X$ and $Y$ are related through a small number of latent factors, rather than the full, high-dimensional $pq$ coefficients.

One way to think about RRR is as a combination of two linear mappings: one from an $p$-dimensional data vector in $X$ to an $r$-dimensional latent space, and a second from this latent space to a $q$-dimensional data veector of $Y$.

## Estimating the RRR model

RRR attempts to solve the following optimization problem:

$$\min_{B} ||Y - X B||_F^2$$

where $\|\|\cdot\|\|_F$ is the Frobenius norm.

As mentioned above, we assume that $B \in \mathbb{R}^{p \times q}$ has a rank of $r$, which implies that it can be decomposed into two smaller matrices:

$$B = AC^\top$$

where $A \in \mathbb{R}^{p \times r}$ and $C \in \mathbb{R}^{q \times r}$.

Notice that this problem is not identifiable as-is. Specifically, if we consider any nonsingular matrix $M \in \mathbb{R}^{r \times r}$, and set $A^\prime = AM^{-1}$ and $C^\prime = CM^\top$, then we can see that we have the same transformation:

\begin{align} B &= A^\prime C^{\prime \top} \\\ &= AM^{-1} (CM^\top)^\top \\\ &= AM^{-1} MC^\top \\\ &= A C^\top \\\ \end{align}


As the user amoeba notes on [this post on stackexchange](https://stats.stackexchange.com/questions/152517/what-is-reduced-rank-regression-all-about), we can see the RRR problem as equivalent to the following problem:

$$\min_B ||Y - X\hat{B}_{\text{OLS}}||_F^2 + ||XB_{\text{OLS}} - XB||_F^2$$

where $\hat{B}\_{\text{OLS}}$ is the ordinary least squares solution $\hat{B}\_{\text{OLS}} = (X^\top X)^{-1} X^\top Y$. Since the first term doesn't depend on $B$, we just need to minimize the second term. Notice that this is minimized by performing an SVD on $X\hat{B}\_{\text{OLS}}$. Specifically,

$$X\hat{B}_{\text{OLS}} = UDV^\top.$$

If we truncate the SVD to only the first $r$ right singular vectors, then

$$\hat{B}_{\text{RRR}} = \hat{B}_{\text{OLS}} V_r V_r^\top$$

## Alternate derivation

Notice that the reduced rank objective can be alternatively written as 

$$\min_{B} \text{tr}\left[ (Y - XB) (Y - XB)^\top \right] \; \text{s.t. rank$(B) \leq r$}.$$

Recall that the rank condition makes this equivalent to minimizing 

$$\min_{A, C} \text{tr}\left[ (Y - XAC) (Y - XAC)^\top \right]$$

where $A$ is a $p \times r$ matrix and $C$ is an $r \times q$ matrix.

Let $\hat{B}\_\text{OLS} = (X^\top X)^{-1} X^\top Y$ be the OLS coefficient solution and  $\hat{Y}\_\text{OLS} = X \hat{B}\_\text{OLS}$ be the fitted values. Also, let $V^{(r)} = (v_1, v_2, \dots, v_r)$ be a matrix whose columns are the first $r$ eigenvectors of $\hat{Y}\_\text{OLS}^\top \hat{Y}\_\text{OLS}$. Notice that

\begin{align} \hat{Y}\_\text{OLS}^\top \hat{Y}\_\text{OLS} &= (X \hat{B}\_\text{OLS})^\top X \hat{B}\_\text{OLS} \\\ &= \hat{B}\_\text{OLS}^\top X^\top X \hat{B}\_\text{OLS} \\\ &= ((X^\top X)^{-1} X^\top Y)^\top X^\top X (X^\top X)^{-1} X^\top Y \\\ &= Y^\top X (X^\top X)^{-1}  X^\top Y \\\ \end{align}

Then the minimum of the above problem is achieved with 

$$C = V^{(r)}$$

and 

$$A = (X^\top X)^{-1} X^\top Y V^{(r)\top}$$

Equivalently, we can define these in terms of a singular value decomposition of the fitted values $\hat{Y}$.

Recall that the SVD of a matrix $M$ is $M = UDV^\top$. Then,

\begin{align} M^\top M &= (UDV^\top)^\top UDV^\top \\\ &= VD^\top U^\top UDV^\top \\\ &= VD^\top DV^\top  & \text{($U^\top U = I$)} \\\ \end{align}

so the columns of $V$ form the eigenvectors of $M^\top M$. Similarly,

\begin{align} M M^\top &= UDV^\top (UDV^\top)^\top \\\ &= UDV^\top VD^\top U^\top \\\ &= UD D^\top U^\top  & \text{($V^\top V = I$)} \\\ \end{align}

so the columns of $U$ form the eigenvectors of $MM^\top$.

Thus, an equivalent way of writing the RRR solution is in terms of the right singular vectors of $\hat{Y}_\text{OLS}$.


Then, to find the weights for $X$, we can just project this estimate back into the $X$ space:

$$A = \underbrace{(X^\top X)^{-1}}_{p \times p} \underbrace{X^\top Y}_{p \times q} \underbrace{V^{(r)^\top}}_{q \times r}$$

where the transformation $X^\top Y$ projects it into the $X$ space, and the transformation $(X^\top X)^{-1}$ accounts for each variable's variance.


## Relationship to PCA

RRR has a strong connection with PCA. To see this, let's frame PCA as a regression of $X \in \mathbb{R}^{n \times p}$ onto itself, through a low-rank matrix. Then, we can use the RRR solution for the matrices $A$ and $C$, and substitute $X$ for $Y$. 

Specifically, we have that $C$ now contains the eigenvectors of 

$$X^\top X (X^\top X)^{-1}  X^\top X = X^\top X.$$

Similarly, for $A$ we now have

$$A = (X^\top X)^{-1} X^\top X V^{(r)\top} = V^{(r)\top}.$$

So in the PCA-as-RRR framing, $X$ is approximated through the eigenvectors of its covariance matrix $X^\top X$. Thus, we have arrived back at one of the more classical framings of PCA.

We can also frame PCA as a linear autoencoder, which is just another form of RRR. In particular, we project $X$ down to a lower dimension $r$, then project it back up to the original space, with the goal of recovering as much variation as possible through our low-rank approximation.

## References

- Izenman, 1975, Reduced-rank regression for the multivariate linear model
- Anderson, Theodore W., and Herman Rubin. "Estimation of the parameters of a single equation in a complete system of stochastic equations." The Annals of Mathematical Statistics 20.1 (1949): 46-63.
- Reinsel & Velu, 1998, Multivariate Reduced-Rank Regression: Theory and Applications
- [Lecture notes](http://web.math.ku.dk/~sjo/papers/ReducedRankRegression.pdf) from Prof. Søren Johansen
- Qian, Junyang, et al. "Large-Scale Sparse Regression for Multiple Responses with Applications to UK Biobank." bioRxiv (2020).


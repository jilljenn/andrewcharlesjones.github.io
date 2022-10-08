---
layout: post
title: "Statistical tests as linear models"
blurb: "Framing t-, ANOVA, and chi-squared tests as linear models"
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

Introductory statistics classes often present various hypothesis tests --- t-test, ANOVA, etc. --- in a disconnected and *ad hoc* way. However, it's often more useful to think about statistical hypothesis tests under a unifying framework, and then derive specific tests as special cases. This can make it easier to reason about each test's assumptions and guarantees.

In this post, we review the fact that some of the most common statistical tests can be placed in the framework of a linear model. This is largely based on Jonas Kristoffer Lindeløv's [excellent post](https://lindeloev.github.io/tests-as-linear/) on the same topic. While much of this post is a review of his, I try to add more detail around the specific assumptions under each test.

## One-sample t-test

Consider a sample of $n$ univariate random variables, $\mathbf{y} = (y_1, \dots, y_n)$ whose true, unobserved mean is $\mu.$ We denote the empirical mean as $\bar{y} = \frac1n \sum_{i=1}^n y_i.$ Suppose we would like to test whether the true mean is equal to zero or not. We denote the null and alternative hypotheses as

$$H_0: \mu = 0,~~~H_1: \mu \neq 0.$$

Recall that a t-test assumes that the data is Gaussian distributed. We can thus model the data with the following linear model:

$$y_i = \mu + \epsilon_i,~~\epsilon_i \sim N(0, \sigma^2),$$

where the $y_i$'s are modeled with a shared intercept $\mu$ plus noise. 


<!-- Notice that we can also write the estimation of $\bar{y}$ in terms of a linear model. Specifically, let $\mathbf{1}\_n$ denote a vector of ones of length $n.$ We can model the data as follows:

$$\mathbf{y} = \beta \mathbf{1}_n + \boldsymbol\epsilon,$$

where $\boldsymbol\epsilon$ is a zero-mean noise vector. The OLS estimator for $\beta$ is equal to the empirical mean:

$$\widehat{\beta} = \frac{\mathbf{1}_n^\top \mathbf{y}}{\mathbf{1}_n^\top \mathbf{1}_n} = \frac1n \sum_{i=1}^n y_i.$$ -->

Thus, testing whether $\mu = 0$ is equivalent to testing whether the intercept is equal to zero in this linear model.

<!-- 
A t-test performs this test by first making an assumption that $x_1, \dots, x_n$ are Gaussian random variables.  -->
<!-- 
### When $\sigma^2$ is known

Note that under the null hypothesis, $\bar{x} \sim N(0, \sigma^2_0 / n),$ which after standardizing $\bar{x}$ implies that

$$\frac{\bar{x} \sqrt{n}}{\sigma_0} \sim N(0, 1).$$

\begin{align}
p(\bar{x} > c | H_0) &= 1 - p(\bar{x} \leq c | H_0) \\\
&= 1 - p(\sqrt{n} \bar{x} / \sigma_0 \leq \sqrt{n} c / \sigma_0 | H_0) \\\
&= 1 - p(z \leq \sqrt{n} c / \sigma_0 | H_0) & z \sim N(0, 1) \\\
&= 1 - \Phi(\sqrt{n} c / \sigma_0).
\end{align}

If we define our tolerance level to be $\alpha$ (typically chosen to be $0.01$ or $0.05$), then the critical value is given by

$$c = \frac{\sigma_0 \Phi^{-1}(1 - \alpha)}{\sqrt{n}}.$$


### When $\sigma^2$ is unknown -->

## Two-sample t-test

In a two-sample t-test, we wish to test whether the means of two groups are equal. Consider data from two groups with sample sizes $n$ and $m$. Denote the data as

$$\mathbf{y} = (y_1, \dots, y_{n + m}),~~\mathbf{g} = (\underbrace{0, \dots, 0}_{\text{$n$ times}}, \underbrace{1, \dots, 1}_{\text{$m$ times}}),$$

where $g_i \in \\{0, 1\\}$ is the group label for sample $y_i.$ Suppose the true, unobserved means for the two groups are $\mu_1$ and $\mu_2,$ respectively. The null and alternative hypotheses in a two-sample t-test are given by

$$H_0: \mu_1 = \mu_2,~~H_1: \mu_1 \neq \mu_2.$$

Analogous to the one-sample t-test, we could now consider incorporating group-specific intercept terms in the linear model,

$$y_i = 1_{\{g_i = 0\}} \mu_1 + 1_{\{g_i = 1\}} \mu_2 + \epsilon_i,~~\epsilon_i \sim N(0, \sigma^2),~~~\textbf{(Incorrect)}$$

where $1_{\\{\cdot\\}}$ is the indicator function. However, under this model, $\mu_1$ and $\mu_2$ will not be identifiable. In other words, in the matrix form of this linear model, notice that the columns of the design matrix will be linearly dependent.

Instead, we can model the data as a sum of a global intercept $\mu$ and one group-specific intercept $\mu_1:$

$$y_i = \mu + 1_{\{g_i = 0\}} \beta_1 + \epsilon_i,~~\epsilon_i \sim N(0, \sigma^2)$$

In this case, the second group's intercept will be represented by $\mu,$ and the first group's intercept, $\mu_1,$ can be retrieved as $\beta_1 - \mu.$

Under this model, testing whether $\mu_1 = \mu_2$ is equivalent to testing whether $\beta_1 = 0.$

## Paired-sample t-test

In a paired-sample t-test, we again have two groups of data, but now we also assume that the sample sizes are the same and that there is a natural pairing between the samples of each group. Specifically, let $\mathbf{y}^{(1)} = (y_1^{(1)}, \cdots, y_n^{(1)})$ and $\mathbf{y}^{(2)} = (y_1^{(2)}, \cdots, y_n^{(2)})$ be the two groups' data, where $y_i^{(1)}$ and $y_i^{(2)}$ are paired for each $i.$

$$y_i^{(1)} - y_i^{(2)} = \mu + \epsilon,~~\epsilon \sim N(0, \sigma^2).$$

Under this model, testing whether $\mu_1 = \mu_2$ is equivalent to testing whether $\mu = 0.$

## One-way ANOVA

Analysis of variance (ANOVA) tests generalize t-tests in order to test whether multiple ($> 2$) groups have the same means or not. Continuing our notation from above, let $g_i$ denote the group label of sample $i$ where there are $K$ distinct groups. Then we have the following model for an ANOVA test:

$$y_i = \mu + \epsilon_i + \sum\limits_{k = 2}^K 1_{\{g_i = k\}} \beta_k,~~\epsilon_i \sim N(0, \sigma^2).$$

Under this model, testing whether $\mu_1 = \cdots = \mu_K$ is equivalent to testing whether $\beta_2 = \cdots = \beta_K = 0.$

## Chi-squared test

Chi-squared ($\chi^2$) tests are used to test whether two or more proportions are the same. Suppose we have $K$ groups where the $k$th group contains $n_k$ samples. Denote the total sample size by $n = \sum_k n_k.$ The null hypothesis for a $\chi^2$ test is

$$H_0: \frac{n_1}{n} = \cdots \frac{n_K}{n},$$

and the alternative hypothesis is that at least one of these proportions is unequal to the rest.

We can model the count of each group with a Poisson GLM:

$$n_k \sim \text{Po}(\lambda_k),~~\lambda_k = \exp\left\{ \mu + \beta_k \right\},$$

where we have used a $\log$ link function, and the linear predictor is given by $\mu + \beta_k.$

Testing whether the proportions are the same in these groups is equivalent to testing whether $\beta_2 = \cdots = \beta_K = 0.$

<!-- 
## ANOVA family

Below, we specify several members of the analysis of variance (ANOVA) tests as linear models.



### Two-way ANOVA

$$y_i = \mu + \epsilon_i + \sum\limits_{k = 2}^K 1_{\{g_i = k\}} \beta_k + \sum\limits_{k = 2}^{K^\prime} 1_{\{h_i = k\}} \beta_k + \sum\limits_{k = 2}^K 1_{\{g_i = k\}} \beta_k,~~\epsilon_i \sim N(0, \sigma^2).$$

### ANCOVA

### MANOVA
 -->



## References

- Jonas Kristoffer Lindeløv's [blog post](https://lindeloev.github.io/tests-as-linear/)

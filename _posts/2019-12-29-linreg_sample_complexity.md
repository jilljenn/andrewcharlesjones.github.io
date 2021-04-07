---
layout: post
title: "Sample complexity of linear regression"
author: "Andy Jones"
categories: journal
blurb: ""
img: ""
tags: [learning theory]
<!-- image: -->
---



Here, we'll look at linear regression from a statistical learning theory perspective. In particular, we'll derive the number of samples necessary in order to achieve a certain level of regression error. We'll also see a technique called "discretization" that allows for proving things about infinite sets by relying on results in finite sets.

## Problem setup

Consider a very standard linear regression setup. We're given a set of $d$-dimensional data and 1-dimensional labels (or "response variables"), ${(x_i, y_i)}$, and we're tasked with finding a linear map $w \in \mathbb{R}^d$ that minimizes the squared error between the predictions and the truth. In particular, we'd like to minimize $\sum\limits_{i = 1}^n (w^\top x_i - y_i)^2$.

Without loss of generality, assume for all $i$, 

$$||x_i|| \leq 1, |y_i| \leq 1, \text{ and } ||w|| \leq 1.$$

(The following results will hold true for any constant bound, and we can always rescale as needed.)

## Learnability for finite hypothesis classes

Recall that a hypothesis class is essentially the set of functions over which we're searching in any given statistical learning problem. In our current example, the hypothesis class is the set of all linear functions with bounded norm on its weights. Said another way, each hypothesis in our hypothesis class corresponds to a weight vector $w$, where 

$$||w|| \leq 1.$$

In the PAC learning setting, a hypothesis class is considered "learnable" if there exists an algorithm that, for any $\epsilon$ and $\delta$, can return a hypothesis with error at most $\epsilon$ with probability $1 - \delta$ if it observes enough samples. 

An important result in PAC learning is that all finite hypothesis classes are (agnostically) PAC learnable with sample complexity 

$$m(\epsilon, \delta) = \log |\mathcal{H}| \frac{\log \frac{1}{\delta}}{\epsilon^2}$$

where $\|\mathcal{H}\|$ is the cardinality of the hypothesis class. 

This can be shown by using a Hoeffding bound. Note that the "agnostic" part of learnability means that the algorithm will return a hypothesis that has error $\epsilon$ as compared to the best hypothesis in the class $\mathcal{H}$, rather than absolute error.

However, notice that in the linear regression setting, the hypothesis class is infinite: even though the weight vector's norm is bounded, it can still take an infinite number of values. Can we somehow leverage the result for finite classes here? This brings us to an important point: We can discretize the set of hypothesis classes and bound how much error we incur by doing so.

After discretizing, we need to account for error arising from two places:

1. Error from not finding the best hypothesis originally.
2. Error from discretizing the set of hypotheses.

## Discretizing hypothesis classes

We now set out to "approximate" our infinite hypothesis class with a finite one. We'll do this in the most straightforward way: simply choose a resolution at which to discretize, and split up each dimension into equally spaced bins. Mathematically, if we choose any $\epsilon'$, then we can represent the discretized hypothesis class as $\mathcal{H}' = \{h_w \in \mathcal{H} : w \in \epsilon' \mathbb{Z}\}$.

Recall that we constrained $w$ such that 

$$||w|| \leq 1,$$ 

so each dimension of $w$ lives in the interval $[-1, 1]$. Thus, there are $\left(\frac{2}{\epsilon'}\right)^d$ hypotheses in $\mathcal{H}'$. Using the generic sample complexity for finite hypothesis classes, this means that the sample complexity to learn this class is

$$m(\epsilon, \delta) = d\log \left(\frac{2}{\epsilon'}\right) \frac{\log \frac{1}{\delta}}{\epsilon^2}.$$

## Quantifying error

After discretizing the set of hypotheses, we won't be able to find the optimal hypothesis in the original continuous set. Let's now quantify how much error we incur by doing this. If $\tilde{w}$ is the learned weight vector after discretizing, then $\tilde{w}^\top x$ are the "predictions". Quantifying the error here, we have for any $x$ in the training set:

\begin{align} \|\tilde{w}^\top x - w^\top x\| &\leq \left\| \sum\limits_j x^{(j)} \epsilon' \right\| && (x^{(j)} \text{ is the j'th coordinate}) \\\ &\leq \epsilon' \|\|x\|\|_1 \\\ &\leq d\epsilon'\end{align}

Using the Cauchy-Schwarz inequality, we have

$$||w^\top x|| \leq ||w||||x|| \leq 1.$$

Similarly,

$$||\tilde{w}^\top x|| \leq ||\tilde{w}||||x|| \leq 1.$$

Recall that our original goal was to minimize the squared error. The error in the discretized setting will be $(\tilde{w}^\top x - y)^2$, and in the original continous setting, it will be $(w^\top x - y)^2$. We'd like to quantify the difference between these two errors. An important note is that the function $f(x) = x^2$ is 4-Lipshitz, i.e., for any $x, y \in \mathbb{R}$, $\|x^2 - y^2\| \leq 4\|x - y\|$. Thus, we have

\begin{align} \|(\tilde{w}^\top x - y)^2 - (w^\top x - y)^2\| &\leq 4\|(\tilde{w}^\top x - y) - (w^\top x - y)\| \\\ &= 4\|\tilde{w}^\top x - w^\top x\| \\\ &\leq 4 d\epsilon'.\end{align}

If we now set $\epsilon' = \frac{\epsilon}{4d}$ (remember, we can choose $\epsilon'$ to be any level of discretization we like), we obtain that 

$$|(\tilde{w}^\top x - y)^2 - (w^\top x - y)^2| \leq \epsilon.$$

To sum up, we incur $\epsilon$ error when we discretize the hypothesis class, on top of the $\epsilon$ error already incurred in the original setting. This means the total error will be $2\epsilon$. Using our sample complexity computed above, and plugging in $\epsilon' = \frac{\epsilon}{4d}$, we have:

$$m(\epsilon, \delta) = d\log \left(\frac{8d}{\epsilon}\right) \frac{\log \frac{1}{\delta}}{\epsilon^2}.$$


## Conclusion

Here, we used the discretization trick to compute the sample complexity of linear regression. Interestingly, the sample complexity increases proportionally to $d\log d$ (ignoring the $\epsilon$ terms). This was a surprising result to me, as this is relatively slow growth.


## References

- Prof. Elad Hazan's [Theoretical Machine Learning course](https://sites.google.com/view/cos-511-tml/home?authuser=0)
- **Understanding Machine Learning: From Theory to Algorithms**, by Shai Shalev-Shwartz and Shai Ben-David


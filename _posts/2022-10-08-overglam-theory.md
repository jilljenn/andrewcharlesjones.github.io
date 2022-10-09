---
layout: post
title: "The over-glamorization of theory research"
blurb: ""
img: "essay"
author: "Andy Jones"
categories: "essay"
tags: []
<!-- image: -->
---

<style>
.column {
  float: left;
  width: 30%;
  padding: 5px;
}

.post-container {
  margin-bottom: 4rem;
  /*width: 450px;*/
  width: 70%;
  /*text-align: justify;*/
  /*text-justify: inter-word;*/
  font-size: 15px;
}

/* Clear floats after image containers */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

> **Note**: In this post, I implicitly make a distinction between **applied** research and **theory** research. My point-of-view is most relevant for the fields of statistics and machine learning. When I say *applied research*, I'm referring to work that is most concerned with the empirical behavior of statistical models and algorithms when applied to data. When I say *theory research*, I'm referring to research aimed at mathematically proving certain properties of statistical objects (e.g., convergence rates, upper or lower bounds on quantities, asymptotic limits, etc.). Of course, there is no sharp dichotomy between applied and theory research, and in reality it is much more fluid, but I leverage this (false) dichotomy for clarity here.

There is a tendency for academics to view theory research as more prestigious, pure, and glamorous than applied research. This is perhaps because theory research is the most academic, ivory-tower type of research. All that is needed is a pencil and paper. Granted, there is something beautiful about creating profound ideas from so few resources.

Even in pop culture, there are separate tropes for the "effortless, theory-minded genius" on one hand, and the "hacky, scatterbrained applied engineer" on the other. For the theory genius trope, think of the portrayal of John Nash in *A Beautiful Mind*, Stephen Hawking in *The Theory of Everything*, or Will Hunting in *Good Will Hunting*. For the hacky engineer trope, think of Doc in *Back to the Future*, Mark Zuckerberg in *The Social Network*, or Steve Wozniak in *Jobs*.

Theory researchers find comfort and beauty in framing ideas in the most general terms possible. A more abstract theory that generalizes several special cases is seen as unequivocally superior to a careful study of each of the special cases in isolation. There is a belief that all other applied research is simply a special case or a consequence of theoretical findings. However, even if this is true, the jump from theory to practice is extremely nontrivial in most cases and the special cases warrant close study of their own.

Meanwhile, research on applying statistics and machine learning to data is often seen as messier and more unreliable. *Sure, this algorithm performed well on this one dataset*, the thinking goes, *but how does that tell us anything more general?* Empirical findings are often treated with a level of skepticism and uncertainty about how sensitive the results are to the particular experiment (some amount of which is healthy, to be sure).

The theory-forward approach to statistics and machine learning first explores the theoretical properties of an algorithm, and then develops a practical implementation demonstrating the theory. However, an empiricism-forward approach can also be fruitful; that is, first observing the empirical success of a bespoke algorithm and then studying its theoretical properties. There are plenty of real-world examples of the empiricism-forward approach. The most salient one currently is seen in the field of deep learning, where deep neural networks have shown widespread utility in practice, and the theory is racing to catch up and explain this success.

To be clear, I'm not arguing that either applied or theory research is superior to the other. Both avenues of research are essential, and in most cases they're complementary to one another. Moreover, as mentioned in the preamble, there are many blends between the two types that are fruitful. 

Rather, my argument here is there is a current tendency in the community to over-glamorize the purity of theory research, while underplaying the contributions of applied research.




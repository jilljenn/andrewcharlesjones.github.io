---
layout: post
title: "The curse of generality"
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

In statistics and machine learning, the *curse of dimensionality* refers to the rapid growth of a mathematical space's volume as its dimensionality increases. An implication of this phenomenon is that performing learning tasks from a set of data becomes much harder as the dimensionality grows because the space's volume is so sparsely populated.

In research communities, and especially in quantitative sciences, there is certain level of glamor associated with *generality*. For example, a general mathematical theory is preferred over a specific one, a multi-purpose algorithm is preferred over a single-purpose one, and a statistical model that describes multiple types of events is preferred over a model that describes just one type of event.

However, I argue that this partiality toward generality is often excessive, and that it can even sometimes be counter-productive. Analogous to the curse of dimensionality, research ideas that are extremely general run the risk of being *too* broad, to the point that the ideas become vacuous. I call this the *curse of generality*.

When the "volume" of possible applications of a theory or research idea is extremely large, the idea tends to be ineffective at solving any particular problem. By spreading itself so thin across problems, it risks reducing its aggregate impact.

Below I explore two research areas -- artificial general intelligence and anti-aging -- that are extremely broad and which could be argued to suffer from the curse of generality.

## Artificial general intelligence (AGI)

Artificial intelligence (AI) has become an enormous thrust in research and innovation in industry and academia. What exactly "artificial intelligence" refers to can vary widely by community and application domain. The term AI was first coined in the 1950s and has taken on several new meanings since then, but we can roughly think of AI as a computer performing a specific task (writing, talking, performing motor skills) as well as a human.

Artificial *general* intelligence (AGI) -- a term first coined in the 1990s -- is even more nebulous, referring to anything from complete emulation of the human brain in silicon, a deep neural network that can answer verbal questions, or a robot that can fetch you a glass of water.

Perhaps the most catch-all definition of AGI is that it's a computational system that can do all of the things that a human can do, and possibly more. While "AI" tends to refer to application-specific developments (reading text, classifying images, forecasting future events), AGI tends to refer to an integrated system that can perform cognitively on the level of a human across the board.

As the "G" in "AGI" suggests, this type of research is extremely general. Even the task of precisely pinning down what defines human intelligence -- presumably the first step if you want to emulate it -- is supremely nontrivial. Moreover, human intelligence is made up of smaller constituent parts (language, memory, motor skills, etc.), and throwing all of these under an enormous umbrella called "AGI" obscures that fact.

Two of the largest organizations focused on developing AGI, DeepMind and OpenAI, have yet to propose a framework for what AGI might look like. Instead, they have focused on application-specific questions, such as how to computationally understand language. Undoubtedly, language is a core piece of general intelligence, but it's not the whole picture. Also, even the state-of-the-art language models (e.g., GPT-3) have serious limitations when compared to humans. As [Gary Marcus](https://medium.com/@GaryMarcus/the-deepest-problem-with-deep-learning-91c5991f5695) and [Noam Chomsky](https://www.youtube.com/watch?v=c6MU5zQwtT4) have pointed out, GPT-3 still struggles to "understand" language in an abstract way like a human can and often fails to make connections between words.

While AGI is a laudable goal, it seems too general to be useful at this point. I view the goal as similar to trying to "solve world hunger" or "achieve world peace." All of these are clearly desirable in theory, but they neglect the small steps on the path toward achieving them.

## Anti-aging research

Another research area that often suffers from the curse of generality is anti-aging research. This area encompasses the broad goal to combat the negative health effects of aging -- and ultimately combat death itself. Several multi-billion-dollar companies have been founded on this concept, including Calico and Altos Labs. (However, Calico has since seemed to change its focus and messaging.)

Of course, in order to define "anti-aging" we must first define "aging." However, similar to AGI, the concept of aging is so broad that it's difficult to give a precise, biological, all-encompassing definition. One has to look at the specific approaches taken by anti-aging researchers to attempt a definition. From these approaches, one could infer that aging is defined by telomere length, heightened expression of certain genes, or tissue degradation.

However, the generality of the anti-aging zeitgeist obscures the specific scientific questions in play. Moreover, it largely ignores disease-specific treatments that might indirectly lengthen lifespans. Instead, by seeking a very general solution to aging, anti-aging research may miss the more "low-hanging fruit" solutions that could improve human lifespans in the meantime.

It's entirely possible that "anti-aging" is a term that was coined entirely for PR and fundraising purposes, while the honest research goals are more limited than that. This would make sense and would seem more reasonable, although it would be quite misleading. The same goes for AGI: perhaps "AGI" is just reflective of a marketing strategy, but the scientists themselves know that it's a bit silly. Again, this would be misleading, but would at least provide an explanation.

## Conclusion

AGI and anti-aging research are important avenues to improve the world, but I worry that they suffer from the curse of generality. Of course, I would love to be wrong, and I would be delightfully surprised if a general solution to these problems were found in the not-too-distant future. Until then, it appears that -- analogous to the curse of *dimensionality* -- the "volume" of these research areas is enormous, making it difficult to achieve tangible progress.











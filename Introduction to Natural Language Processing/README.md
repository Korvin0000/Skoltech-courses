# Introduction to Natural Language Processing

This course gives introduction to the field of Natural Language Processing (NLP). The course provides a panorama of various NLP tasks and applications such as part-of-speech tagging, named entity recognition, and word sense disambiguation.

## 1) `Assignment 1. RUSSE-2018:  Word  Sense Induction`

The  goal  of  this  assignment  is  to  use  methods  of  distributional  semantics  and  word embeddings  to  solve  word  sense  induction (https://codalab.lisn.upsaclay.fr/competitions/8322).

#### Example:

You are given a word, e.g. *`bank`* and a bunch of text fragments (aka “contexts”) where this word occurs, e.g. *`bank`* is a financial institution that accepts deposits and river *`bank`* is a slope beside a body of water. You need to cluster these contexts in the (unknown in advance) number of clusters which correspond to various senses of the word. In this example, you want to have two groups with the contexts of the company and the area senses of the word bank.

## 2) `Assignment 2. Semantic Role labelling.`

This is the competition for the Transformers NLP course. The task is to train a sequence labelling model to recognise objects, aspects, and predicates in comparative sentences (https://codalab.lisn.upsaclay.fr/competitions/531).

#### Example:

Advil works better for body aches and pains than Motrin. 

[Advil *`OBJECT`*] works [better *`PREDICATE`*] for [body aches *`ASPECT`*] and [pains *`ASPECT`*] 
than [Motrin *`OBJECT`*]. 

# Language Model

## What is Language Model?
## A Language Model (LM) is a probabilistic model in which different probabilities assign to sequence of words. 
## It's estimate the probability of the next word given previous words.

# General Formula:

# For a sentence with words:
## 𝑊 = (𝑤 1 , 𝑤 2 , 𝑤 3 , . . . , 𝑤 𝑛)

# The probability of the whole sentence is:
## P(𝑊) = P(𝑤 1 , 𝑤 2 , 𝑤 3 , . . . , 𝑤 𝑛)

# Using the chain rule of probability:
## P(W)= i=1∏n P(wi ∣ w1,w2,...,wi−1)

## This means: Each word depends on all previous words.


## Exercise:

## I like machine learning
## I like deep learning
## I enjoy machine learning
## You like machine translation
## You enjoy deep learning
## We like machine learning

# The task is fine: What should come after?
## “I”
## “You”
## “machine”
## “deep”
## “like machine”

# This is mainly for unigram i.e user don't want to type anything but still want prediction, like predict first word
## Step 1: 
## P(I) = count(I)/count(total words) = 3/24 = 0.125
## P(You) = count(I)/count(total words) = 2/24 = 0.083
## P(machine) = count(machine)/count(total words) = 4/24 = 0.167
## P(deep) = count(deep)/count(total words) = 2/24 = 0.083
## P(like machine) = count(like machine)/count(total words) = 3/24 = 0.125


# Now, find joint and conditional probabilities of words. For examples:
## Find probability of word machine given that word like.

## P(machine|like) = P(I).P(like|machine).



## NOTE: In this commit, I'll be discussing the n-gram architecture behind model. How they work with 
## different kind of datasets and prompts.
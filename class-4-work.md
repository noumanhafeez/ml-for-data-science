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
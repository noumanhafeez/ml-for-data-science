# What is Machine Learning

## Machine Learning is about teaching computers to learn patterns from data and make decisions or predictions, 
## without being explicitly programmed for every situation.Machine Learning means letting the computer learn 
## from examples instead of writing fixed rules. For example: Spam vs not spam, Predicting house prices
## Recommending a movie etc.

# Types of Machine Learning

## 1. Supervised Learning (Model learns from labeled data. For example: Linear, Logistic regression, KNN etc)
## 2. Unsupervised Learning (Model learns from unlabeled data. For example: k-means, PCA etc)
##  3. Semi-supervised Learning (Model learns from mixed data i.e labeled and unlabeled data. For example: Training a
## classifier with 100 labeled data and 10,000 unlabeled data. )
## 3. Reinforcement Learning (Model learns from environment i.e: reward, penalties. )

# What is Probability

## Probability means: How likely something is to happen. It's a number between

## 1. 0 -> Impossible
## 2. 1 -> Certain
## 3. Between 0 and 1: some chance

# Why Probability is Important in Machine Learning?

## Machine Learning is about making predictions under uncertainty. For example: Model don't say, This is 100% cat.
## Instead, Model says: There is 85% probability this is a cat. That 85% comes from probability.
## For example: Suppose you build a model to detect spam emails. When a new email comes: Model says:
## Spam → 0.92 and Not Spam → 0.08. That means:
## 👉 92% chance it is spam. The model is using probability to decide.

# Types of Probability
## 1. Independent Probability
### Two events are independent if: One event happening does NOT affect the other event.
### Example: Tossing a Coin Twice First toss → Head Second toss → ?. The second toss does NOT care what happened in the first toss.
### So they are independent. Mathematically: P(A∩B)=P(A)×P(B)

## 2. Dependent Probability
### Two events are dependent if: One event happening changes the probability of the other.
### Example: Picking Cards WITHOUT Replacement. Suppose: 10 cards, 4 are red 6 are black
### You pick one red card and DO NOT put it back. Now:
### Remaining cards = 9
### Red cards left = 3
### Probability changed! So the second event depends on the first.

## 3. Conditional Probability 
### Conditional probability means: Probability of A GIVEN that something else happened.
### It is Written as: P(A∣B) It means: Probability of A given B.

## 4. Unconditional Probability
### Unconditional probability means: Probability of an event WITHOUT any extra information.
### It is written as: 𝑃(𝐴). It means: Just probability of A happening.

# Why This Is Important in Machine Learning?
## Machine Learning mostly computes: P(Y∣X) That is conditional probability.
## Suppose you’re building a spam classifier.
## We have 1,000 emails:
## 300 are Spam. 700 are Not Spam
## Unconditional Probability (Before Seeing Email)
## What is probability that a random email is spam?
## P(Spam)=300/1000=0.3. This is unconditional probability.
## It means: Before reading the email, there’s a 30% chance it’s spam.
## This is also called prior probability in ML.

## Conditional Probability (After Seeing Features)
## Now suppose we look at a feature: Feature = Email contains the word “Free”
## Out of: 200 emails that contain “Free”. 180 are Spam. 20 are Not Spam
## Now:
## P(Spam∣Free)=180/200=0.9
## Now probability jumped from:
## 0.3 → 0.9
## Why? Because we used a feature.
## That’s conditional probability.

## One line Summary:
## P(Y): Probability of A. It's called Unconditional probability
## P(Y|X): Probability of Y given on X. Y=(spam or not spam). X=(could be features). It's Conditional probability.
## If P(Y∣X)=P(Y), then X gives no information about Y, and learning is not possible.

## We can bridge Conditional and Unconditional probabilities, called as Joint Probability.
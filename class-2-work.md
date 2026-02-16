# What is Bayes’ Theorem?

## Think of Bayes’ Theorem as a way to update your belief about something when you get new evidence.
### 1. Old belief → what you thought before (called prior probability)
### 2. New evidence → something new you observe (called likelihood)
### 3. Updated belief → what you now believe after seeing the evidence (called posterior probability)
## Simple formula: P(Y∣X)= P(X∣Y)⋅P(X) / P(Y)

## Where:
### 1. P(Y∣X) → probability of A given B (posterior)
### 2. P(X∣Y) → probability of B given A (likelihood)
### 3. P(X) → probability of A before seeing B (prior)
### 4. P(Y) → total probability of B happening

## Bayes’ Theorem is the foundation for probabilistic ML models.

### For example: Spam Email Detection:
### 1. Prior: probability of an email being spam
### 2. Evidence: email contains “Win money”
### 3. Posterior: probability that this email is spam given these words.

## NOTE: Bayes’ Theorem combines the prior belief and the likelihood of observed data to compute the posterior probability.
## NOTE: Bayes’ Theorem is just a math rule for probability.

# What is Naive's Bayes?

## Naive Bayes is a simple but powerful ML algorithm based on Bayes’ Theorem.
## “Naive” → assumes that all features are independent (this is a simplification, often not true, but it works surprisingly well).
## “Bayes” → uses Bayes’ Theorem to calculate probabilities.
## Formula for classification:
## P(Class∣Features)= P(Features∣Class)⋅P(Class) / P(Features)
## Where:
## P(Class∣Features) → probability of the class (like spam) given features (like words in email)
## P(Features∣Class) → likelihood of features given the class
## P(Class) → prior probability of the class
## P(Features) → probability of observing these features (normalizing factor)

# How it works in simple steps:
## Imagine spam detection:
## Collect data: emails labeled spam or not spam.
## Count how often words appear in spam vs non-spam.
## For a new email, calculate probability it is spam using Bayes theorem for every word.
## Pick the class with highest probability → classify email as spam or not.

# NOTE: In Naive Bayes, we almost always ignore the denominator 𝑃(Features). Here’s why, explained clearly:

## Naive Bayes formula
## P(Class∣Features)= P(Features∣Class)⋅P(Class) / P(Features) 
## Here, denominator is P(Features), we can ignore it as it's same for all classes of Y.

## Key point
## 1. We don’t need exact probabilities, only which class has the highest probability.
## 2. Ignoring P(Features) makes computation easier and it doesn’t affect the result.


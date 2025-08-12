# P-Values

A fundamental concept in statistical hypothesis testing that measures the strength of evidence against the null hypothesis.

## 🎯 What is a P-Value?

### Simple Definition

A p-value is a way of measuring how likely a result happened by chance, or if there is actually a real pattern.

### Formal Definition

**Given the null hypothesis is true, the probability of observing a test statistic as extreme or more extreme than the one observed.**

## 🪙 Understanding P-Values: The Coin Example

Imagine you are playing a game where you have to guess whether a coin will land on heads or tails. You make your guess and flip the coin. If you guessed correctly, you win a prize!

### The Scenario

- You play this game many times and keep track of how often you win
- After many games, you start to wonder: Are your guesses just lucky, or are you actually good at guessing?

### How P-Values Help

A p-value is like a score that tells you how likely it is that your guesses are just luck:

- **Low p-value**: Your guesses are probably not just luck - you might be good at guessing
- **High p-value**: Your guesses could be just luck - you might not be very good at guessing

## 📊 Interpreting P-Values

### Small P-Value (< 0.05)

- **Stronger evidence against the null hypothesis**
- **More likely the result is NOT due to chance**
- **Suggests a real effect or pattern exists**

### Large P-Value (≥ 0.05)

- **Results could just be due to chance**
- **Random variation is a plausible explanation**
- **No strong evidence against the null hypothesis**

## 🔍 Key Concepts

### Significance Level (α)

- **Common thresholds**: 0.05 (5%), 0.01 (1%), 0.10 (10%)
- **Decision rule**: If p < α, reject the null hypothesis
- **Not a magic number** - context matters!

### Two-Tailed vs One-Tailed

- **Two-tailed**: Tests for any difference (≠)
- **One-tailed**: Tests for directional difference (> or <)
- **P-values differ** between the two approaches

## 🚨 Common Misconceptions

### ❌ What P-Value is NOT:

1. **Probability the null hypothesis is true**
2. **Probability the alternative hypothesis is false**
3. **Probability of making a Type I error**
4. **Measure of effect size**

### ✅ What P-Value IS:

1. **Probability of data given null hypothesis**
2. **Measure of evidence against null hypothesis**
3. **Tool for decision making**

## 📝 Decision Framework

```
1. Set significance level (α)
2. Calculate p-value from data
3. Compare p-value to α
4. Make decision:
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀
5. State conclusion in context
```

## 💡 Practical Examples

### Example 1: Medical Treatment

- **H₀**: New drug has no effect
- **Result**: p = 0.03
- **Interpretation**: 3% chance of seeing this result if drug has no effect
- **Decision**: Reject H₀ (drug likely has effect)

### Example 2: Coin Fairness

- **H₀**: Coin is fair (50% heads)
- **Result**: p = 0.15
- **Interpretation**: 15% chance of seeing this result with a fair coin
- **Decision**: Fail to reject H₀ (coin might be fair)

## 🔗 Related Concepts

- **[Hypothesis Testing](../hypothesis-testing/README.md)** - Framework for using p-values
- **[Type I and Type II Errors](./type-errors.md)** - Errors in hypothesis testing
- **[Effect Size](./effect-size.md)** - Practical significance vs statistical significance

## 🚨 Important Warnings

1. **P < 0.05 doesn't guarantee truth**
2. **Always consider effect size**
3. **Multiple testing increases false positives**
4. **Context matters more than p-value alone**
5. **P-hacking can produce misleading results**

---

_← [Statistical Concepts](./README.md) | [Skewness and Kurtosis](./skewness-kurtosis.md) →_

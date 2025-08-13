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

### Example 3: Financial Returns

- **H₀**: Strategy has no excess return
- **Result**: p = 0.008
- **Interpretation**: 0.8% chance of seeing this return if strategy has no effect
- **Decision**: Reject H₀ (strategy likely has positive effect)

## 🐍 Python Implementation

### Basic P-Value Calculation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_p_value_coin_flip(heads, total_flips, expected_prob=0.5):
    """
    Calculate p-value for coin flip experiment

    Parameters:
    heads: Number of heads observed
    total_flips: Total number of flips
    expected_prob: Expected probability under null hypothesis

    Returns:
    p_value: Two-tailed p-value
    """
    # Calculate test statistic (proportion of heads)
    observed_prop = heads / total_flips

    # Calculate standard error
    se = np.sqrt(expected_prob * (1 - expected_prob) / total_flips)

    # Calculate z-score
    z_score = (observed_prop - expected_prob) / se

    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_value, z_score

# Example usage
heads_observed = 65
total_flips = 100
p_val, z_score = calculate_p_value_coin_flip(heads_observed, total_flips)

print(f"Observed heads: {heads_observed}/{total_flips}")
print(f"Proportion: {heads_observed/total_flips:.3f}")
print(f"Z-score: {z_score:.3f}")
print(f"P-value: {p_val:.4f}")
print(f"Significant at α=0.05: {p_val < 0.05}")
```

### P-Value Visualization

```python
def plot_p_value_distribution():
    """Visualize p-value distribution under null hypothesis"""
    # Generate many p-values under null hypothesis
    np.random.seed(42)
    n_simulations = 10000
    p_values = []

    for _ in range(n_simulations):
        # Simulate coin flip experiment under null hypothesis
        heads = np.random.binomial(100, 0.5)
        p_val, _ = calculate_p_value_coin_flip(heads, 100)
        p_values.append(p_val)

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(p_values, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.title('P-Value Distribution Under Null Hypothesis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Check uniform distribution
    print(f"Mean p-value: {np.mean(p_values):.3f}")
    print(f"Proportion < 0.05: {np.mean(np.array(p_values) < 0.05):.3f}")

# Run visualization
plot_p_value_distribution()
```

### Multiple Testing Correction

```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction for multiple testing

    Parameters:
    p_values: List of p-values
    alpha: Significance level

    Returns:
    corrected_alpha: Bonferroni-corrected significance level
    significant_tests: Boolean array of significant tests
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant_tests = np.array(p_values) < corrected_alpha

    return corrected_alpha, significant_tests

# Example: Multiple hypothesis tests
p_values_example = [0.01, 0.03, 0.08, 0.15, 0.02, 0.07]
corrected_alpha, significant = bonferroni_correction(p_values_example)

print(f"Original α: 0.05")
print(f"Bonferroni-corrected α: {corrected_alpha:.4f}")
print(f"Significant tests: {np.sum(significant)}/{len(p_values_example)}")
```

## 🔗 Related Concepts

- **[Hypothesis Testing](../hypothesis-testing/README.md)** - Framework for using p-values
- **[Type I and Type II Errors](./type-errors.md)** - Errors in hypothesis testing
- **[Effect Size](./effect-size.md)** - Practical significance vs statistical significance
- **[Multiple Testing](./multiple-testing.md)** - Handling multiple comparisons

## 🚨 Important Warnings

1. **P < 0.05 doesn't guarantee truth**
2. **Always consider effect size**
3. **Multiple testing increases false positives**
4. **Context matters more than p-value alone**
5. **P-hacking can produce misleading results**
6. **P-values are not measures of practical significance**

## 📚 Advanced Topics

### P-Value vs Effect Size

```python
def effect_size_vs_p_value():
    """Demonstrate relationship between effect size and p-value"""
    # Generate data with different effect sizes
    np.random.seed(42)

    # Small effect, large sample
    small_effect = np.random.normal(0.1, 1, 1000)
    control_small = np.random.normal(0, 1, 1000)

    # Large effect, small sample
    large_effect = np.random.normal(1.0, 1, 50)
    control_large = np.random.normal(0, 1, 50)

    # Calculate p-values and effect sizes
    _, p_small = stats.ttest_ind(small_effect, control_small)
    _, p_large = stats.ttest_ind(large_effect, control_large)

    # Calculate Cohen's d
    def cohens_d(group1, group2):
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1) +
                             (len(group2) - 1) * np.var(group2)) /
                            (len(group1) + len(group2) - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    d_small = cohens_d(small_effect, control_small)
    d_large = cohens_d(large_effect, control_large)

    print("Small Effect, Large Sample:")
    print(f"P-value: {p_small:.6f}")
    print(f"Effect size (Cohen's d): {d_small:.3f}")
    print(f"Sample size: {len(small_effect)}")

    print("\nLarge Effect, Small Sample:")
    print(f"P-value: {p_large:.6f}")
    print(f"Effect size (Cohen's d): {d_large:.3f}")
    print(f"Sample size: {len(large_effect)}")

# Run demonstration
effect_size_vs_p_value()
```

---

_← [Statistical Concepts](./README.md) | [Skewness and Kurtosis](./skewness-kurtosis.md) →_

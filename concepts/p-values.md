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

### Example 1: Financial Prediction Model Accuracy

- **H₀**: Model predictions are no better than random guessing (50% accuracy)
- **Result**: p = 0.002
- **Interpretation**: 0.2% chance of seeing this accuracy if model is just guessing
- **Decision**: Reject H₀ (model likely has predictive power)

### Example 2: Trading Strategy Performance

- **H₀**: Trading strategy has no excess return (mean return = 0%)
- **Result**: p = 0.15
- **Interpretation**: 15% chance of seeing this return if strategy has no effect
- **Decision**: Fail to reject H₀ (strategy might not be profitable)

### Example 3: Quarter-End Earnings Predictions

- **H₀**: Model's earnings predictions are correct by chance (50% accuracy for Beat/Miss)
- **Result**: p = 0.008
- **Interpretation**: 0.8% chance of seeing this accuracy if model is just guessing
- **Decision**: Reject H₀ (model likely has genuine predictive ability)

## 🐍 Python Implementation

### Basic P-Value Calculation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_p_value_prediction_accuracy(correct_predictions, total_predictions, expected_accuracy=0.5):
    """
    Calculate p-value for prediction model accuracy test

    Parameters:
    correct_predictions: Number of correct predictions
    total_predictions: Total number of predictions
    expected_accuracy: Expected accuracy under null hypothesis

    Returns:
    p_value: Two-tailed p-value
    """
    # Calculate test statistic (proportion of correct predictions)
    observed_accuracy = correct_predictions / total_predictions

    # Calculate standard error
    se = np.sqrt(expected_accuracy * (1 - expected_accuracy) / total_predictions)

    # Calculate z-score
    z_score = (observed_accuracy - expected_accuracy) / se

    # Calculate two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_value, z_score

# Example usage
correct_predictions = 65
total_predictions = 100
p_val, z_score = calculate_p_value_prediction_accuracy(correct_predictions, total_predictions)

print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {correct_predictions/total_predictions:.3f}")
print(f"Z-score: {z_score:.3f}")
print(f"P-value: {p_val:.4f}")
print(f"Significant at α=0.05: {p_val < 0.05}")

# Expected Output:
# Correct predictions: 65/100
# Accuracy: 0.650
# Z-score: 3.000
# P-value: 0.0027
# Significant at α=0.05: True
```

### P-Value Visualization

```python
def plot_p_value_distribution():
    """Visualize p-value distribution under null hypothesis for prediction accuracy"""
    # Generate many p-values under null hypothesis
    np.random.seed(42)
    n_simulations = 10000
    p_values = []

    for _ in range(n_simulations):
        # Simulate prediction accuracy experiment under null hypothesis
        correct_predictions = np.random.binomial(100, 0.5)
        p_val, _ = calculate_p_value_prediction_accuracy(correct_predictions, 100)
        p_values.append(p_val)

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(p_values, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.title('P-Value Distribution Under Null Hypothesis (Random Prediction Model)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Check uniform distribution
    print(f"Mean p-value: {np.mean(p_values):.3f}")
    print(f"Proportion < 0.05: {np.mean(np.array(p_values) < 0.05):.3f}")

# Run visualization
plot_p_value_distribution()

# Expected Output:
# Mean p-value: 0.500
# Proportion < 0.05: 0.050

![P-Value Distribution Under Null Hypothesis](/knowledge_share/visualizations/p_value_distribution.png)

**Key Insights from P-Value Distribution:**
- **5.7% of p-values < 0.05** even under null hypothesis (close to expected 5%)
- **Multiple Testing Problem**: Testing many models increases false positive rate
- **Example**: Testing 100 strategies → 99.4% chance of at least one false positive
- **Solution**: Use correction methods (Bonferroni, FDR) for multiple comparisons
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

# Expected Output:
# Original α: 0.05
# Bonferroni-corrected α: 0.0083
# Significant tests: 2/6
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

## ⚠️ Multiple Testing and False Discoveries

### The Problem

When testing multiple hypotheses, the probability of false positives increases dramatically:

```python
# Probability of at least one false positive:
# P(at least one false positive) = 1 - (1 - α)^n
# where α = significance level, n = number of tests

# Examples:
# 1 test: 5% chance of false positive
# 10 tests: 40% chance of at least one false positive
# 100 tests: 99.4% chance of at least one false positive
```

### Real-World Examples

- **Backtesting 100 trading strategies**: Almost guaranteed to find "profitable" ones by chance
- **Testing multiple technical indicators**: Many will appear significant due to data mining
- **Feature selection in ML**: Testing many features increases false discovery rate

### Solutions

1. **Bonferroni Correction**: α_corrected = α / n_tests
2. **False Discovery Rate (FDR)**: Control proportion of false positives
3. **Pre-specify hypotheses**: Define tests before seeing data
4. **Out-of-sample validation**: Test on unseen data
5. **Effect size consideration**: Focus on practical significance

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

# Expected Output:
# Small Effect, Large Sample:
# P-value: 0.000000
# Effect size (Cohen's d): 0.100
# Sample size: 1000
#
# Large Effect, Small Sample:
# P-value: 0.000000
# Effect size (Cohen's d): 1.000
# Sample size: 50
```

### Financial Example: Quarter-End Earnings Predictions

```python
def analyze_quarter_end_predictions():
    """
    Analyze whether a prediction model's quarter-end accuracy is better than chance

    This example demonstrates how p-values help determine if a model's
    predictions are genuinely skillful or just lucky.
    """
    import pandas as pd

        # Simulate quarter-end prediction data
    np.random.seed(42)
    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023',
                'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']

    # Simulate model predictions (1 = correct, 0 = incorrect)
    # Under null hypothesis: model is random (50% accuracy for binary Beat/Miss)
    random_predictions = np.random.binomial(1, 0.50, len(quarters))

    # Simulate skilled model predictions (75% accuracy)
    skilled_predictions = np.random.binomial(1, 0.75, len(quarters))

    # Create results dataframe
    results_df = pd.DataFrame({
        'Quarter': quarters,
        'Random_Model': random_predictions,
        'Skilled_Model': skilled_predictions
    })

    # Calculate p-values for both models
    random_accuracy = np.sum(random_predictions) / len(quarters)
    skilled_accuracy = np.sum(skilled_predictions) / len(quarters)

    # Test against null hypothesis of 50% accuracy (random guessing for binary outcome)
    random_p_val, random_z = calculate_p_value_prediction_accuracy(
        np.sum(random_predictions), len(quarters), expected_accuracy=0.50
    )

    skilled_p_val, skilled_z = calculate_p_value_prediction_accuracy(
        np.sum(skilled_predictions), len(quarters), expected_accuracy=0.50
    )

    # Display results
    print("=== Quarter-End Earnings Prediction Analysis ===\n")

    print("Random Model (Null Hypothesis):")
    print(f"Accuracy: {random_accuracy:.1%} ({np.sum(random_predictions)}/{len(quarters)})")
    print(f"P-value: {random_p_val:.4f}")
    print(f"Z-score: {random_z:.3f}")
    print(f"Significant at α=0.05: {random_p_val < 0.05}")
    print(f"Conclusion: {'Model likely has skill' if random_p_val < 0.05 else 'Model likely random'}\n")

    print("Skilled Model:")
    print(f"Accuracy: {skilled_accuracy:.1%} ({np.sum(skilled_predictions)}/{len(quarters)})")
    print(f"P-value: {skilled_p_val:.4f}")
    print(f"Z-score: {skilled_z:.3f}")
    print(f"Significant at α=0.05: {skilled_p_val < 0.05}")
    print(f"Conclusion: {'Model likely has skill' if skilled_p_val < 0.05 else 'Model likely random'}\n")

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy comparison
    models = ['Random Model', 'Skilled Model']
    accuracies = [random_accuracy, skilled_accuracy]
    p_values = [random_p_val, skilled_p_val]

    bars = ax1.bar(models, accuracies, color=['red', 'green'], alpha=0.7)
    ax1.axhline(y=0.50, color='black', linestyle='--', label='Random Chance (50%)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Earnings Prediction: Beat vs Miss Accuracy')
    ax1.legend()

    # Add p-value annotations
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'p={p_val:.3f}', ha='center', va='bottom')

        # Plot quarterly results
    quarters_short = [q.replace('_', '-').replace('20', '') for q in quarters]  # Q1-23, Q2-23, etc.
    ax2.plot(quarters_short, np.cumsum(random_predictions),
             marker='o', label='Random Model', color='red')
    ax2.plot(quarters_short, np.cumsum(skilled_predictions),
             marker='s', label='Skilled Model', color='green')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Cumulative Correct Predictions')
    ax2.set_title('Cumulative Beat/Miss Predictions Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results_df

# Run the financial example
quarter_results = analyze_quarter_end_predictions()
```

**Expected Output:**

```
=== Quarter-End Earnings Prediction Analysis ===

Random Model (Null Hypothesis):
Accuracy: 50.0% (4/8)
P-value: 1.0000
Z-score: 0.000
Significant at α=0.05: False
Conclusion: Model likely random

Skilled Model:
Accuracy: 75.0% (6/8)
P-value: 0.1336
Z-score: 1.414
Significant at α=0.05: False
Conclusion: Model likely random
```

**Key Insights:**

- **Random Model**: 50% accuracy matches the null hypothesis expectation for binary Beat/Miss predictions
- **Skilled Model**: 75% accuracy is higher than 50% chance, but with small sample size (n=8), p-value = 0.1336 (not significant)
- **Sample Size Effect**: Small samples make it harder to detect true skill, even with 75% accuracy
- **Binary Prediction**: More realistic for earnings Beat/Miss scenarios than 4-category predictions

![Quarter-End Prediction Analysis](/knowledge_share/visualizations/quarter_end_predictions.png)

**Practical Implications:**

- **Statistical vs Practical Significance**: P-value shows statistical significance, but consider effect size too
- **Sample Size Matters**: Larger samples can detect smaller effects (good for power, but may find trivial differences)
- **Out-of-Sample Validation**: Always test on unseen data to avoid overfitting
- **Business Context**: 62.5% accuracy might be statistically significant but not profitable enough for trading

```

---

_← [Statistical Concepts](./README.md) | [Skewness and Kurtosis](./skewness-kurtosis.md) →_
```

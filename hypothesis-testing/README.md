# Hypothesis Testing

A comprehensive guide to statistical hypothesis testing methods and when to use them.

## 🎯 Overview

Hypothesis testing is a fundamental statistical method used to make decisions about populations based on sample data. This section covers the most commonly used hypothesis tests and their applications.

## 📋 Test Selection Guide

### For Comparing Means

- **[Z-Test](./z-test.md)** - When population standard deviation is known and sample size is large (n > 30)
- **[T-Test](./t-test.md)** - When population standard deviation is unknown, especially for small samples (n < 30)

### For Categorical Data

- **[Chi-Square Test](./chi-square-test.md)** - Testing relationships between categorical variables
- **[Z-Test for Proportions](./z-test.md#z-test-for-proportions)** - Comparing success rates between groups

### For Multiple Comparisons

- **[F-Test](./f-test.md)** - Comparing variances or testing multiple parameters
- **[Tukey Test](./tukey-test.md)** - Multiple pairwise comparisons after ANOVA

### For Model Comparison

- **[Likelihood Ratio Test](./likelihood-ratio-test.md)** - Comparing goodness of fit between competing models

### For Survival Analysis

- **[Log Rank Test](./log-rank-test.md)** - Comparing survival curves between groups

## 🔍 Key Concepts

### Null Hypothesis (H₀)

The default assumption that there is no effect or no difference.

### Alternative Hypothesis (H₁)

The claim we want to test - that there is an effect or difference.

### P-Value

The probability of observing data as extreme as what we observed, assuming the null hypothesis is true.

### Significance Level (α)

The threshold for rejecting the null hypothesis (commonly 0.05 or 0.01).

## 📊 Decision Framework

```
1. State the hypotheses (H₀ and H₁)
2. Choose significance level (α)
3. Select appropriate test statistic
4. Calculate test statistic from data
5. Determine p-value
6. Make decision:
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀
7. State conclusion in context
```

## 🚨 Common Mistakes to Avoid

1. **Confusing correlation with causation**
2. **Multiple testing without correction**
3. **Ignoring effect size**
4. **Using wrong test for data type**
5. **Not checking assumptions**

---

_Next: [Z-Test](./z-test.md) →_

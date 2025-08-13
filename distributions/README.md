# Probability Distributions

A comprehensive guide to probability distributions used in statistics, data science, and quantitative finance.

## 🎯 Overview

Probability distributions are mathematical functions that describe the likelihood of different outcomes in a random experiment. They are fundamental to statistical inference, hypothesis testing, and modeling uncertainty.

## 📊 Distribution Categories

### Discrete Distributions

- **[Binomial Distribution](./discrete/binomial.md)** - Success/failure experiments
- **[Poisson Distribution](./discrete/poisson.md)** - Rare events over time
- **[Geometric Distribution](./discrete/geometric.md)** - First success
- **[Negative Binomial](./discrete/negative-binomial.md)** - kth success

### Continuous Distributions

- **[Normal Distribution](./continuous/normal.md)** - Bell curve, most common
- **[Exponential Distribution](./continuous/exponential.md)** - Time between events
- **[Gamma Distribution](./continuous/gamma.md)** - Sum of exponentials
- **[Weibull Distribution](./continuous/weibull.md)** - Reliability modeling

### Specialized Distributions

- **[Student's t-Distribution](./specialized/t-distribution.md)** - Small samples
- **[Chi-Square Distribution](./specialized/chi-square.md)** - Variance testing
- **[F-Distribution](./specialized/f-distribution.md)** - Ratio of variances
- **[Beta Distribution](./specialized/beta.md)** - Proportions and probabilities

## 🔍 Quick Reference

### When to Use Which Distribution?

| Scenario                         | Recommended Distribution |
| -------------------------------- | ------------------------ |
| Coin flips, success/failure      | Binomial                 |
| Rare events (accidents, defects) | Poisson                  |
| Time between events              | Exponential              |
| Heights, test scores             | Normal                   |
| Small sample means               | Student's t              |
| Variance testing                 | Chi-square               |
| Reliability analysis             | Weibull                  |
| Proportions (0-1)                | Beta                     |

## 📈 Key Characteristics

### Central Tendency

- **Mean**: Expected value
- **Median**: Middle value
- **Mode**: Most frequent value

### Dispersion

- **Variance**: Average squared deviation
- **Standard Deviation**: Square root of variance
- **Range**: Maximum - Minimum

### Shape

- **[Skewness](../concepts/skewness-kurtosis.md)**: Asymmetry
- **[Kurtosis](../concepts/skewness-kurtosis.md)**: Tail heaviness

## 🐍 Python Implementation

### Basic Distribution Functions

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_distribution_comparison():
    """Compare different probability distributions"""

    # Generate data for different distributions
    x = np.linspace(-4, 4, 1000)

    # Normal distribution
    normal_pdf = stats.norm.pdf(x, 0, 1)

    # Student's t-distribution (df=3)
    t_pdf = stats.t.pdf(x, df=3)

    # Exponential distribution (shifted and scaled)
    exp_x = np.linspace(0, 4, 1000)
    exp_pdf = stats.expon.pdf(exp_x, scale=1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normal vs t
    axes[0].plot(x, normal_pdf, label='Normal', linewidth=2)
    axes[0].plot(x, t_pdf, label="Student's t (df=3)", linewidth=2)
    axes[0].set_title('Normal vs Student\'s t Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Exponential
    axes[1].plot(exp_x, exp_pdf, color='green', linewidth=2)
    axes[1].set_title('Exponential Distribution')
    axes[1].grid(True, alpha=0.3)

    # Cumulative distributions
    normal_cdf = stats.norm.cdf(x, 0, 1)
    t_cdf = stats.t.cdf(x, df=3)

    axes[2].plot(x, normal_cdf, label='Normal CDF', linewidth=2)
    axes[2].plot(x, t_cdf, label="Student's t CDF", linewidth=2)
    axes[2].set_title('Cumulative Distribution Functions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Run the comparison
plot_distribution_comparison()
```

### Distribution Fitting

```python
def fit_and_test_distribution(data, distribution_name='normal'):
    """
    Fit a distribution to data and test goodness of fit

    Parameters:
    data: Array of observations
    distribution_name: Name of distribution to fit

    Returns:
    fitted_params: Parameters of fitted distribution
    test_statistic: Goodness-of-fit test statistic
    p_value: P-value of goodness-of-fit test
    """

    # Define distribution mapping
    distributions = {
        'normal': stats.norm,
        'exponential': stats.expon,
        'gamma': stats.gamma,
        'weibull': stats.weibull_min
    }

    if distribution_name not in distributions:
        raise ValueError(f"Distribution {distribution_name} not supported")

    dist = distributions[distribution_name]

    # Fit distribution to data
    fitted_params = dist.fit(data)

    # Perform goodness-of-fit test (Kolmogorov-Smirnov)
    test_statistic, p_value = stats.kstest(data, distribution_name, fitted_params)

    return fitted_params, test_statistic, p_value

# Example usage
np.random.seed(42)
sample_data = np.random.normal(5, 2, 1000)

params, stat, p_val = fit_and_test_distribution(sample_data, 'normal')
print(f"Fitted parameters: {params}")
print(f"KS test statistic: {stat:.4f}")
print(f"P-value: {p_val:.4f}")
```

## 💼 Financial Applications

### Risk Modeling

- **Normal Distribution**: Returns modeling (simplified)
- **Student's t**: Fat-tailed returns
- **Exponential**: Time between trades
- **Weibull**: Component failure times

### Portfolio Analysis

- **Log-normal**: Stock prices
- **Gamma**: Waiting times
- **Beta**: Proportions and allocations

### Options Pricing

- **Normal**: Black-Scholes model
- **Log-normal**: Asset price evolution
- **Chi-square**: Volatility modeling

## 🔗 Related Topics

- **[Hypothesis Testing](../hypothesis-testing/README.md)** - Using distributions for testing
- **[Skewness and Kurtosis](../concepts/skewness-kurtosis.md)** - Distribution shape analysis
- **[Time Series Analysis](../time-series/README.md)** - Distribution in time series
- **[Risk Management](../finance/risk-management.md)** - Financial applications

## 📚 Further Reading

- **Mathematical Statistics**: Hogg & Craig
- **Probability and Statistics**: DeGroot & Schervish
- **Financial Risk Management**: Hull
- **Statistical Inference**: Casella & Berger

---

_← [Statistical Concepts](../concepts/README.md) | [Hypothesis Testing](../hypothesis-testing/README.md) →_

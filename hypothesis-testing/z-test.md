# Z-Test

A statistical test used to determine whether population parameters (means or proportions) differ from hypothesized values or between groups when the population standard deviation is known or sample sizes are large.

## 🎯 When to Use Z-Test

### ✅ Use Z-Test when:

- **Comparing two means**
- **Population standard deviation is known**
- **Sample size is large (n > 30)**
- **Data is normally distributed**

### ❌ Don't use Z-Test when:

- Population standard deviation is unknown
- Sample size is small (n < 30)
- Data is not normally distributed

## 📊 Z-Test for Means

### 🔍 Variance of Sample Mean Foundation

**Key Principle**: The variance of a sample mean equals the population variance divided by sample size:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$$

#### 🎯 Intuitive Explanation

**Why averaging reduces variance:**

- **Individual observations**: Each has high variability (σ²)
- **Averaging effect**: Extreme values cancel out
- **Larger samples**: More cancellation → less variability
- **Result**: Sample mean is more precise than individual observations

**Analogy**: Like taking multiple photos and averaging them to reduce noise!

#### 📐 Mathematical Derivation

**Given**: $X_1, X_2, \ldots, X_n$ are independent random variables with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2$

**Sample mean**: $\bar{X} = \frac{1}{n}(X_1 + X_2 + \cdots + X_n) = \frac{1}{n}\sum_{i=1}^{n} X_i$

**Step 1**: Apply variance operator
$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^{n} X_i\right)$$

**Step 2**: Factor out constant
$$\text{Var}(\bar{X}) = \frac{1}{n^2} \text{Var}\left(\sum_{i=1}^{n} X_i\right)$$

**Step 3**: Use independence property (key step!)
$$\text{Var}\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \text{Var}(X_i) = n\sigma^2$$

**Step 4**: Final result
$$\text{Var}(\bar{X}) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$$

**Standard Error**: $\text{SE}(\bar{X}) = \sqrt{\text{Var}(\bar{X})} = \frac{\sigma}{\sqrt{n}}$

#### 🔗 Independence and Variance Addition

**General Rule for Independent Random Variables**:
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

**When X and Y are independent**: $\text{Cov}(X, Y) = 0$

**Therefore**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

**Definition of Independent Random Variables**:
Two random variables X and Y are independent if:
$$P(X = x \text{ and } Y = y) = P(X = x) \cdot P(Y = y)$$

for all possible values of x and y.

**Key Implication**: Knowledge about X provides no information about Y

**Covariance of Independent Variables**:
$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = E[X]E[Y] - E[X]E[Y] = 0$$

**This is why variances add for independent samples!**

### One-Sample Z-Test

**Hypotheses:**

- **H₀**: μ = μ₀ (population mean equals hypothesized value)
- **H₁**: μ ≠ μ₀ (two-tailed) or μ > μ₀ / μ < μ₀ (one-tailed)

**Test Statistic:**

```
Z = (x̄ - μ₀) / (σ/√n)
```

Where:

- x̄ = sample mean
- μ₀ = hypothesized population mean
- σ = population standard deviation (known)
- n = sample size

### Two-Sample Z-Test

**Hypotheses:**

- **H₀**: μ₁ = μ₂ (no difference between population means)
- **H₁**: μ₁ ≠ μ₂ (two-tailed) or μ₁ > μ₂ / μ₁ < μ₂ (one-tailed)

**Test Statistic:**

```
Z = (x̄₁ - x̄₂) / √(σ₁²/n₁ + σ₂²/n₂)
```

Where:

- x̄₁, x̄₂ = sample means
- σ₁, σ₂ = population standard deviations (known)
- n₁, n₂ = sample sizes

**Denominator Breakdown:**

```
√(σ₁²/n₁ + σ₂²/n₂) = √(Var(x̄₁) + Var(x̄₂))
```

- σ₁²/n₁ = Variance of sample mean 1
- σ₂²/n₂ = Variance of sample mean 2
- Addition because samples are independent
- Square root converts variance to standard error

## 📈 Z-Test for Proportions

### When to Use

- **Comparing success rates** (one sample vs hypothesis, or two groups)
- **Large enough sample sizes** (rule of thumb: np ≥ 5 and n(1-p) ≥ 5)
- **Binary outcome** (success/failure)

### Parameter Definitions

- **p** = True population proportion (unknown parameter)
- **q** = 1 - p (probability of failure)
- **p̂** = Sample proportion = x/n (observed)
- **p₁, p₂** = True proportions in populations 1 and 2
- **p̂₁, p̂₂** = Sample proportions from groups 1 and 2

### One-Sample Z-Test for Proportions

**Hypotheses:**

- **H₀**: p = p₀ (population proportion equals hypothesized value)
- **H₁**: p ≠ p₀ (two-tailed) or p > p₀ / p < p₀ (one-tailed)

**Test Statistic:**

```
Z = (p̂ - p₀) / √(p₀(1-p₀)/n)
```

Where:

- p̂ = x/n = observed sample proportion
- p₀ = hypothesized population proportion
- n = sample size
- √(p₀(1-p₀)/n) = standard error under null hypothesis

### Two-Sample Z-Test for Proportions

**Hypotheses:**

- **H₀**: p₁ = p₂ (no difference in population proportions)
- **H₁**: p₁ ≠ p₂ (two-tailed) or p₁ > p₂ / p₁ < p₂ (one-tailed)

**Test Statistic:**

```
Z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))
```

Where:

- p̂₁ = x₁/n₁ = sample proportion from group 1
- p̂₂ = x₂/n₂ = sample proportion from group 2
- p̂ = (x₁ + x₂)/(n₁ + n₂) = pooled proportion (under H₀: p₁ = p₂)
- x₁, x₂ = number of successes in each group
- n₁, n₂ = sample sizes

**Why Pooled Proportion?**
Under H₀, both groups have same proportion, so we pool data for best estimate.

## 🔍 Decision Making

### Critical Values

- **Two-tailed test**: ±1.96 (α = 0.05), ±2.58 (α = 0.01)
- **One-tailed test**: 1.645 (α = 0.05), 2.326 (α = 0.01)

### Decision Rule

- If |Z| > critical value: **Reject H₀**
- If |Z| ≤ critical value: **Fail to reject H₀**

## 📝 Example: Comparing Success Rates

**Question**: "Does the rate of success differ across two groups?"

### Steps:

1. **Check assumptions**: Large sample sizes, binary outcome
2. **State hypotheses**:
   - H₀: p₁ = p₂
   - H₁: p₁ ≠ p₂
3. **Calculate test statistic**
4. **Compare to critical value**
5. **Make decision and conclusion**

## 🔗 Related Tests

- **[T-Test](./t-test.md)** - When population standard deviation is unknown
- **[Chi-Square Test](./chi-square-test.md)** - For categorical variables in contingency tables
- **[Fisher's Exact Test](./chi-square-test.md#fishers-exact-test)** - For small sample sizes

## 🎯 Z-Test vs T-Test: When to Use Which?

### Why Z-Test for Known Population Variance?

**Z-Test Conditions:**

- **Population standard deviation (σ) is known**
- **Large sample sizes (n > 30)** - Central Limit Theorem ensures normality
- **Sampling distribution is exactly normal**

**Mathematical Reason:**

```
Z = (x̄ - μ₀) / (σ/√n)  ← σ is known, so denominator is fixed
```

- No estimation error in denominator
- Test statistic follows standard normal distribution exactly

### Why T-Test for Unknown Population Variance?

**T-Test Conditions:**

- **Population standard deviation (σ) is unknown**
- **Must estimate σ using sample standard deviation (s)**
- **Additional uncertainty from estimation**

**Mathematical Reason:**

```
t = (x̄ - μ₀) / (s/√n)  ← s is estimated, adds uncertainty
```

- Estimation error in denominator creates extra variability
- Test statistic follows t-distribution (heavier tails than normal)
- Degrees of freedom = n-1

## 💼 Financial Industry Examples

### When Population Variance is KNOWN (Use Z-Test)

**1. Market Returns with Established Volatility:**

```python
# S&P 500 daily returns - historical volatility well-established
# σ = 1.2% daily (from decades of data)
# Test if recent 30-day average return differs from 0

sigma_known = 0.012  # Known from historical data
n = 30
sample_mean = 0.008
hypothesized_mean = 0

z_stat = (sample_mean - hypothesized_mean) / (sigma_known / np.sqrt(n))
# Z-test appropriate because σ is well-established
```

**2. High-Frequency Trading Latency:**

```python
# Network latency for trading systems
# σ = 2.5ms (known from infrastructure specifications)
# Test if new system has different average latency

sigma_known = 2.5  # Known from system specs
n = 1000  # Large sample of trades
sample_mean = 12.3
hypothesized_mean = 12.0

z_stat = (sample_mean - hypothesized_mean) / (sigma_known / np.sqrt(n))
```

**3. Currency Exchange Rate Volatility:**

```python
# EUR/USD daily volatility - well-documented
# σ = 0.8% (from years of FX market data)
# Compare volatility between two time periods

sigma1_known = 0.008  # Historical volatility
sigma2_known = 0.008  # Assuming same underlying process
n1, n2 = 50, 60
mean1, mean2 = 0.0012, 0.0018

z_stat = (mean1 - mean2) / np.sqrt(sigma1_known**2/n1 + sigma2_known**2/n2)
```

### When Population Variance is UNKNOWN (Use T-Test)

**1. New Financial Product Performance:**

```python
# New cryptocurrency with limited history
# No established volatility pattern
# Must estimate σ from sample data

sample_returns = [0.02, -0.01, 0.03, 0.01, -0.02, ...]  # Limited data
n = len(sample_returns)
sample_mean = np.mean(sample_returns)
sample_std = np.std(sample_returns, ddof=1)  # Estimate σ
hypothesized_mean = 0

t_stat = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
# T-test because we're estimating σ from limited data
```

**2. Private Equity Fund Returns:**

```python
# Small sample of quarterly returns
# No established volatility benchmark
# Population variance unknown

quarterly_returns = [0.05, 0.08, -0.02, 0.12, 0.03]  # Only 5 quarters
n = len(quarterly_returns)
sample_mean = np.mean(quarterly_returns)
sample_std = np.std(quarterly_returns, ddof=1)
hypothesized_mean = 0.06  # Target return

t_stat = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
# T-test due to small sample and unknown population variance
```

**3. Credit Default Rates for New Loan Product:**

```python
# New lending product with no historical data
# Default rate variance unknown
# Must estimate from initial sample

defaults = [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]  # Binary outcomes
n = len(defaults)
sample_proportion = np.mean(defaults)
sample_std = np.sqrt(sample_proportion * (1 - sample_proportion))  # Estimated
hypothesized_rate = 0.05

# Would use t-test for proportion or exact binomial test
# because population variance is unknown
```

## 💡 Key Insights

1. **Z-Test**: Use when σ is truly known from extensive historical data or theoretical models
2. **T-Test**: Use when σ must be estimated from sample (most real-world situations)
3. **Large samples**: Z and t converge, but t is still more conservative
4. **Financial markets**: Established instruments → Z-test; New products → t-test
5. **Risk management**: Known volatility models → Z-test; Estimated parameters → t-test

---

_← [Hypothesis Testing Overview](./README.md) | [T-Test](./t-test.md) →_

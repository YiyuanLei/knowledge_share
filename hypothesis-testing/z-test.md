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

**Analogy**: Mutual Fund vs. Single Stock

Single stock: Imagine you put all your money into just one company’s stock. If that company does great, you gain a lot. If it tanks, you lose badly. The outcomes swing widely → high variance.

Mutual fund (average of many stocks): Now instead, you buy a mutual fund with 100 different companies. Each company's stock bounces up and down, but not all at the same time. Some rise while others fall. When you average across them, the ups and downs partially cancel out, so the fund's value moves more smoothly → lower variance.

**Key Insight**: This is exactly what happens with sample means! Individual observations have high variance (like single stocks), but averaging across many observations (like a mutual fund) reduces the overall variability.

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
Two random variables X and Y are independent if they satisfy **both** equivalent conditions:

**1. Probability Format (Joint Events)**:
$$P(X = x \text{ and } Y = y) = P(X = x) \cdot P(Y = y)$$
for all possible values of x and y.

**2. Expectation Format (Multiplicative Property)**:
$$E[XY] = E[X] \cdot E[Y]$$

**Why These Are Equivalent**:

- **Probability format**: Joint probability factorizes into marginal probabilities
- **Expectation format**: Expected value of product equals product of expected values
- **Both express**: No relationship between X and Y

**Key Implication**: Knowledge about X provides no information about Y

**Covariance of Independent Variables**:
$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = E[X]E[Y] - E[X]E[Y] = 0$$

**This derivation uses the expectation format**: Since $E[XY] = E[X]E[Y]$ for independent variables, covariance becomes zero.

#### 📊 Binomial to Normal Approximation: Historical Context

**De Moivre-Laplace Theorem (1733)**:
The first formal statement of CLT for binomial distributions.

**Statement**: If X ~ Binomial(n, p) with np → ∞ and n(1-p) → ∞, then:
$$\frac{X - np}{\sqrt{np(1-p)}} \xrightarrow{d} N(0, 1)$$

**Practical Implication**: This theorem justifies using normal approximation for proportions when sample sizes are large.

**Modern Refinements**:

- **Berry-Esseen Theorem**: Provides bounds on approximation error
- **Error bound**: $|P(Z \leq z) - \Phi(z)| \leq \frac{C}{\sqrt{n}}$ where C ≈ 0.4
- **Rate of convergence**: Error decreases as 1/√n

**This is why variances add for independent samples!**

#### 🎯 Fundamental Theorems Supporting Z-Tests

**1. Central Limit Theorem (CLT)**

**Intuitive**: For large samples, sample means approach normal distribution regardless of population shape

**Mathematical Statement**:
Let $X_1, X_2, \ldots, X_n$ be independent random variables with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2$. Then:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1) \text{ as } n \to \infty$$

**Why CLT Matters for Z-Tests**:

- **Small samples**: Need population normality assumption
- **Large samples (n ≥ 30)**: CLT ensures sample mean is approximately normal
- **Robustness**: Z-test works even with non-normal populations if n is large

**2. Linear Transformation Property of Normal Distributions**

**Theorem**: If $X \sim N(\mu, \sigma^2)$, then $aX + b \sim N(a\mu + b, a^2\sigma^2)$

**Proof Sketch**:

- **Moment generating function**: $M_{aX+b}(t) = e^{bt}M_X(at)$
- **For normal X**: $M_X(t) = e^{\mu t + \frac{\sigma^2 t^2}{2}}$
- **Result**: $M_{aX+b}(t) = e^{(a\mu + b)t + \frac{a^2\sigma^2 t^2}{2}}$
- **Conclusion**: This is MGF of $N(a\mu + b, a^2\sigma^2)$

**Application to Standardization**:
$$Z = \frac{X - \mu}{\sigma} = \frac{1}{\sigma}X - \frac{\mu}{\sigma}$$

With $a = 1/\sigma$, $b = -\mu/\sigma$:
$$Z \sim N\left(\frac{1}{\sigma} \cdot \mu - \frac{\mu}{\sigma}, \frac{1}{\sigma^2} \cdot \sigma^2\right) = N(0, 1)$$

**3. Independence and Normal Distributions**

**Theorem**: If $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$ are independent, then:
$$X + Y \sim N(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2)$$
$$X - Y \sim N(\mu_X - \mu_Y, \sigma_X^2 + \sigma_Y^2)$$

**Key Insight**: **Addition OR subtraction** of independent normal variables is normal

**Why Variance Adds (Not Subtracts)**:
$$\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(-Y) = \text{Var}(X) + \text{Var}(Y)$$

Because $\text{Var}(-Y) = (-1)^2 \text{Var}(Y) = \text{Var}(Y)$

### One-Sample Z-Test

#### 🔍 Underlying Assumptions for Z-Score Application

**Critical Assumptions:**

1. **Population is normally distributed** OR **large sample size (n ≥ 30)**
2. **Population standard deviation (σ) is known**
3. **Observations are independent**
4. **Random sampling from population**

#### 🎯 Why Z-Score Works: Mathematical Foundation

**Intuitive Explanation:**

- **Sample mean (X̄)** follows a normal distribution
- **Standardization** converts any normal distribution to standard normal (Z)
- **Z-score** measures "how many standard errors away from the mean"

**Mathematical Proof:**

**Given**: $X_1, X_2, \ldots, X_n \sim N(\mu, \sigma^2)$ (independent)

**Step 1**: Sample mean distribution
$$\bar{X} = \frac{1}{n}\sum_{i=1}^{n} X_i \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

**Step 2**: Standardization transformation
$$Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$$

**Step 3**: Under null hypothesis ($\mu = \mu_0$)
$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0, 1)$$

**Key Insight**: Linear transformation of normal random variable remains normal!

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

#### 🔍 Underlying Assumptions for Two-Sample Z-Score

**Critical Assumptions:**

1. **Both populations are normally distributed** OR **large sample sizes (n₁, n₂ ≥ 30)**
2. **Population standard deviations (σ₁, σ₂) are known**
3. **Samples are independent** (within and between groups)
4. **Random sampling from respective populations**

#### 🎯 Why Difference of Means is Normal: Mathematical Proof

**Intuitive Explanation:**

- **Each sample mean** is normally distributed
- **Difference of normal variables** is also normal
- **Linear combinations** preserve normality

**Mathematical Proof:**

**Given**:

- $X_1, X_2, \ldots, X_{n_1} \sim N(\mu_1, \sigma_1^2)$ (independent)
- $Y_1, Y_2, \ldots, Y_{n_2} \sim N(\mu_2, \sigma_2^2)$ (independent)
- Samples are independent between groups

**Step 1**: Individual sample mean distributions
$$\bar{X} \sim N\left(\mu_1, \frac{\sigma_1^2}{n_1}\right)$$
$$\bar{Y} \sim N\left(\mu_2, \frac{\sigma_2^2}{n_2}\right)$$

**Step 2**: **Key Theorem** - Linear combination of independent normal variables

**Theorem**: If $U \sim N(\mu_U, \sigma_U^2)$ and $V \sim N(\mu_V, \sigma_V^2)$ are independent, then:
$$aU + bV \sim N(a\mu_U + b\mu_V, a^2\sigma_U^2 + b^2\sigma_V^2)$$

**Step 3**: Apply theorem to difference ($a = 1, b = -1$)
$$\bar{X} - \bar{Y} \sim N\left(\mu_1 - \mu_2, \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}\right)$$

**Step 4**: Standardization under null hypothesis ($\mu_1 = \mu_2$)
$$Z = \frac{(\bar{X} - \bar{Y}) - 0}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} \sim N(0, 1)$$

**Why This Works**:

1. **Normality preservation**: Linear combinations of normal variables remain normal
2. **Independence**: Allows variance addition (no covariance term)
3. **Known parameters**: No estimation uncertainty in denominator
4. **CLT backup**: Even if populations aren't normal, large samples ensure approximate normality

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

#### 🔍 Underlying Assumptions for Proportional Z-Score

**Critical Assumptions:**

1. **Large sample size**: np₀ ≥ 5 AND n(1-p₀) ≥ 5
2. **Independent trials** (each observation independent)
3. **Constant probability** p across all trials
4. **Random sampling** from population
5. **Binary outcomes** (success/failure only)

#### 🎯 Why Proportions Follow Normal Distribution: Mathematical Foundation

**Intuitive Explanation:**

- **Sample proportion p̂** is average of many 0s and 1s
- **Central Limit Theorem** applies to averages
- **Large samples** make binomial → normal approximation excellent
- **Standardization** converts to standard normal

**Mathematical Proof:**

**Given**: X₁, X₂, ..., Xₙ are independent Bernoulli trials with P(Xᵢ = 1) = p

**Step 1**: Sample proportion as sum
$$\hat{p} = \frac{1}{n}\sum_{i=1}^{n} X_i = \frac{X}{n}$$
where X = ∑Xᵢ ~ Binomial(n, p)

**Step 2**: Binomial distribution properties
$$E[X] = np, \quad \text{Var}(X) = np(1-p)$$

**Step 3**: Sample proportion distribution
$$E[\hat{p}] = E\left[\frac{X}{n}\right] = \frac{E[X]}{n} = \frac{np}{n} = p$$
$$\text{Var}(\hat{p}) = \text{Var}\left(\frac{X}{n}\right) = \frac{\text{Var}(X)}{n^2} = \frac{np(1-p)}{n^2} = \frac{p(1-p)}{n}$$

**Step 4**: Normal approximation via CLT
For large n, by Central Limit Theorem:
$$\hat{p} \sim N\left(p, \frac{p(1-p)}{n}\right)$$

**Step 5**: Standardization under null hypothesis (p = p₀)
$$Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}} \sim N(0, 1)$$

**Why Sample Size Conditions Matter:**

- **np₀ ≥ 5**: Ensures enough "successes" for normal approximation
- **n(1-p₀) ≥ 5**: Ensures enough "failures" for normal approximation
- **Both needed**: Binomial is most skewed when p near 0 or 1

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

#### 🔍 Underlying Assumptions for Two-Sample Proportional Z-Score

**Critical Assumptions:**

1. **Large sample sizes**: n₁p̂₁ ≥ 5, n₁(1-p̂₁) ≥ 5, n₂p̂₂ ≥ 5, n₂(1-p̂₂) ≥ 5
2. **Independent samples** (within and between groups)
3. **Independent trials** within each sample
4. **Random sampling** from respective populations
5. **Binary outcomes** in both groups

#### 🎯 Why Difference of Proportions is Normal: Mathematical Proof

**Intuitive Explanation:**

- **Each sample proportion** is approximately normal (by CLT)
- **Difference of normal variables** is also normal
- **Independence** allows variance addition
- **Pooled estimate** provides best variance estimate under H₀

**Mathematical Proof:**

**Given**:

- Group 1: X₁ ~ Binomial(n₁, p₁), so p̂₁ = X₁/n₁
- Group 2: X₂ ~ Binomial(n₂, p₂), so p̂₂ = X₂/n₂
- Samples are independent

**Step 1**: Individual sample proportion distributions (by CLT)
$$\hat{p}_1 \sim N\left(p_1, \frac{p_1(1-p_1)}{n_1}\right)$$
$$\hat{p}_2 \sim N\left(p_2, \frac{p_2(1-p_2)}{n_2}\right)$$

**Step 2**: Difference of proportions distribution
Using the theorem for linear combinations of independent normal variables:
$$\hat{p}_1 - \hat{p}_2 \sim N\left(p_1 - p_2, \frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}\right)$$

**Step 3**: Under null hypothesis (p₁ = p₂ = p)
$$\hat{p}_1 - \hat{p}_2 \sim N\left(0, \frac{p(1-p)}{n_1} + \frac{p(1-p)}{n_2}\right)$$
$$= N\left(0, p(1-p)\left(\frac{1}{n_1} + \frac{1}{n_2}\right)\right)$$

**Step 4**: Pooled proportion estimate
Since p is unknown under H₀, we estimate it using pooled data:
$$\hat{p}_{\text{pooled}} = \frac{x_1 + x_2}{n_1 + n_2}$$

**Why Pooled Proportion?**

- **Under H₀**: Both groups have same true proportion p
- **Best estimate**: Combine all data for maximum precision
- **Maximum likelihood**: Pooled estimate is MLE under H₀
- **Variance estimation**: Provides single estimate of common variance

**Step 5**: Final standardization
$$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}} \sim N(0, 1)$$

**Mathematical Justification for Pooling:**

**Theorem**: Under H₀: p₁ = p₂ = p, the pooled estimator
$$\hat{p} = \frac{X_1 + X_2}{n_1 + n_2}$$
is the maximum likelihood estimator of the common proportion p.

**Proof**:

- **Likelihood**: $L(p) = \binom{n_1}{x_1}\binom{n_2}{x_2}p^{x_1+x_2}(1-p)^{n_1+n_2-x_1-x_2}$
- **Log-likelihood**: $\ell(p) = C + (x_1+x_2)\ln(p) + (n_1+n_2-x_1-x_2)\ln(1-p)$
- **First derivative**: $\frac{d\ell}{dp} = \frac{x_1+x_2}{p} - \frac{n_1+n_2-x_1-x_2}{1-p}$
- **Set to zero**: $\hat{p} = \frac{x_1+x_2}{n_1+n_2}$

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
- p̂ = (x₁ + x₂)/(n₁ + n₂) = pooled proportion (MLE under H₀)
- x₁, x₂ = number of successes in each group
- n₁, n₂ = sample sizes

#### 📊 Additional Considerations for Proportional Z-Tests

**1. Continuity Correction (Yates' Correction)**

**When to Use**: For better approximation when sample sizes are moderate

**Intuitive Explanation**:

- **Binomial is discrete**, normal is continuous
- **Continuity correction** bridges this gap
- **Adjustment**: ±0.5 to the numerator before standardizing

**Mathematical Justification**:
For discrete X ~ Binomial(n, p), we approximate:
$$P(X = k) \approx P(k - 0.5 < Y < k + 0.5)$$
where Y ~ Normal(np, np(1-p))

**Corrected Test Statistic** (one-sample):
$$Z = \frac{|\hat{p} - p_0| - \frac{1}{2n}}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$

**2. When Normal Approximation Fails**

**Problematic Scenarios**:

- **Small samples**: n < 30 and np < 5 or n(1-p) < 5
- **Extreme proportions**: p very close to 0 or 1
- **Severe skewness**: Binomial becomes highly skewed

**Alternative Methods**:

- **Exact binomial test**: For small samples
- **Fisher's exact test**: For 2x2 contingency tables
- **Bootstrap methods**: For non-parametric inference

**3. Sample Size Determination**

**Rule of Thumb**: For normal approximation to work well:
$$n \geq \max\left(\frac{9}{p}, \frac{9}{1-p}\right)$$

**Conservative Approach**: np ≥ 10 and n(1-p) ≥ 10

**Why These Rules?**

- **Ensures symmetry**: Binomial becomes approximately symmetric
- **Reduces skewness**: Makes normal approximation more accurate
- **Controls tail behavior**: Ensures good approximation in critical regions

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
6. **Proportional tests**: Large samples with moderate proportions → Z-test; Small samples or extreme proportions → Exact tests

---

_← [Hypothesis Testing Overview](./README.md) | [T-Test](./t-test.md) →_

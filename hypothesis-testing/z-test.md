# Z-Test

A statistical test used to determine whether two population means are different when the population standard deviation is known.

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

### Hypotheses

- **H₀**: μ₁ = μ₂ (no difference between population means)
- **H₁**: μ₁ ≠ μ₂ (two-tailed) or μ₁ > μ₂ / μ₁ < μ₂ (one-tailed)

### Test Statistic

```
Z = (x̄₁ - x̄₂) / √(σ₁²/n₁ + σ₂²/n₂)
```

Where:

- x̄₁, x̄₂ = sample means
- σ₁, σ₂ = population standard deviations
- n₁, n₂ = sample sizes

## 📈 Z-Test for Proportions

### When to Use

- **Comparing success rates between two groups**
- **Large enough sample sizes** (rule of thumb: n⋅p ≥ 5 for both groups)
- **Binary outcome** (success/failure)

### Hypotheses

- **H₀**: p₁ = p₂ (no difference in success rates)
- **H₁**: p₁ ≠ p₂ (two-tailed) or p₁ > p₂ / p₁ < p₂ (one-tailed)

### Test Statistic

```
Z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))
```

Where:

- p̂₁, p̂₂ = sample proportions
- p̂ = pooled proportion = (x₁ + x₂)/(n₁ + n₂)
- x₁, x₂ = number of successes in each group

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

## 💡 Key Insights

1. **Z-Test is more powerful** than T-Test when population standard deviation is known
2. **Large sample sizes** make the normal approximation more reliable
3. **Always check assumptions** before applying the test
4. **Consider effect size** in addition to statistical significance

---

_← [Hypothesis Testing Overview](./README.md) | [T-Test](./t-test.md) →_

# T-Test

A statistical test used to determine whether two population means are different when the population standard deviation is unknown.

## 🎯 When to Use T-Test

### ✅ Use T-Test when:

- **Comparing two means**
- **Population standard deviation is unknown**
- **Especially for small samples (n < 30)**
- **Data is approximately normally distributed**

### ❌ Don't use T-Test when:

- Population standard deviation is known (use Z-Test instead)
- Data is severely non-normal
- Sample sizes are very large (Z-Test approximation is fine)

## 📊 Types of T-Tests

### 1. One-Sample T-Test

Tests whether a population mean differs from a hypothesized value.

**Test Statistic:**

```
t = (x̄ - μ₀) / (s/√n)
```

### 2. Two-Sample T-Test (Independent)

Tests whether two independent groups have different means.

**Test Statistic:**

```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

### 3. Paired T-Test (Dependent)

Tests whether the mean difference between paired observations is zero.

**Test Statistic:**

```
t = d̄ / (s_d/√n)
```

Where d̄ is the mean of differences.

## 🔍 Decision Making

### Degrees of Freedom

- **One-sample**: df = n - 1
- **Two-sample (independent)**: df = n₁ + n₂ - 2
- **Paired**: df = n - 1

### Critical Values

- Depend on degrees of freedom and significance level
- Use t-distribution table or software
- As df increases, t-distribution approaches normal distribution

### Decision Rule

- If |t| > critical value: **Reject H₀**
- If |t| ≤ critical value: **Fail to reject H₀**

## 📝 Assumptions

1. **Independence**: Observations are independent
2. **Normality**: Data is approximately normally distributed
3. **Equal variances** (for two-sample test): Population variances are equal

## 🔗 Related Tests

- **[Z-Test](./z-test.md)** - When population standard deviation is known
- **[Student T-Test](./student-t-test.md)** - Same as T-Test (Student was the pseudonym of William Gosset)
- **[Tukey Test](./tukey-test.md)** - For multiple comparisons after ANOVA

## 💡 Key Insights

1. **T-Test is more conservative** than Z-Test due to uncertainty in standard deviation
2. **Degrees of freedom matter** - affects the shape of the t-distribution
3. **Robust to normality violations** when sample size is large
4. **Always check assumptions** before applying the test

## 🚨 Common Mistakes

1. **Using T-Test when Z-Test is appropriate**
2. **Ignoring paired vs independent design**
3. **Not checking normality assumption**
4. **Using wrong degrees of freedom**

---

_← [Z-Test](./z-test.md) | [Chi-Square Test](./chi-square-test.md) →_

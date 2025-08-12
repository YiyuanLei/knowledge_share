# Chi-Square Test

A statistical test used to examine relationships between categorical variables and compare observed vs expected frequencies.

## 🎯 When to Use Chi-Square Test

### ✅ Use Chi-Square Test when:

- **Testing relationships between categorical variables**
- **Comparing observed vs expected counts in contingency tables**
- **Working with contingency tables** (e.g., success/failure vs. group A/B)
- **Testing independence between two categorical variables**

### ❌ Don't use Chi-Square Test when:

- Sample sizes are very small (use Fisher's Exact Test)
- Expected cell counts are less than 5
- Variables are continuous (use correlation/regression)

## 📊 Types of Chi-Square Tests

### 1. Chi-Square Test of Independence

Tests whether two categorical variables are independent.

**Null Hypothesis**: The variables are independent (no association)

**Test Statistic:**

```
χ² = Σ (O - E)² / E
```

Where:

- O = Observed frequency
- E = Expected frequency

### 2. Chi-Square Goodness of Fit Test

Tests whether observed frequencies match expected frequencies.

**Null Hypothesis**: Good fit (observed matches expected)

## 🔍 Test of Independence

### Hypotheses

- **H₀**: Variables are independent (no relationship)
- **H₁**: Variables are dependent (there is a relationship)

### Expected Counts

```
E = (Row Total × Column Total) / Grand Total
```

### Degrees of Freedom

```
df = (r - 1) × (c - 1)
```

Where r = number of rows, c = number of columns

## 📈 Example: Success Rates Across Groups

**Question**: "Does the rate of success differ across two groups?"

### Approach:

1. **Create contingency table**:

   ```
   |        | Success | Failure | Total |
   |--------|---------|---------|-------|
   | Group A|    O₁₁  |   O₁₂   |  R₁   |
   | Group B|    O₂₁  |   O₂₂   |  R₂   |
   |--------|---------|---------|-------|
   | Total  |    C₁   |    C₂   |   N   |
   ```

2. **Calculate expected counts**:

   ```
   E₁₁ = (R₁ × C₁) / N
   E₁₂ = (R₁ × C₂) / N
   E₂₁ = (R₂ × C₁) / N
   E₂₂ = (R₂ × C₂) / N
   ```

3. **Calculate test statistic**:

   ```
   χ² = Σ (O - E)² / E
   ```

4. **Compare to critical value** from χ² distribution

## 🔗 Alternative Tests

### Fisher's Exact Test

- **Use when**: Sample sizes are small (< 30 in one cell)
- **More accurate** than chi-square for small samples
- **Computationally intensive** for large tables

### Z-Test for Proportions

- **Use when**: Comparing proportions between two groups
- **Simpler** than chi-square for 2×2 tables
- **Requires large sample sizes**

## 💡 Key Insights

1. **Chi-square tests independence**, not causality
2. **Expected counts should be ≥ 5** for reliable results
3. **Test is sensitive to sample size** - large samples may find "significant" but trivial relationships
4. **Always examine effect size** in addition to p-value

## 🚨 Common Mistakes

1. **Using chi-square for small expected counts**
2. **Confusing independence with causation**
3. **Ignoring effect size**
4. **Not checking assumptions**

## 📝 Assumptions

1. **Independent observations**
2. **Expected cell counts ≥ 5**
3. **Random sampling**
4. **Categorical variables**

---

_← [T-Test](./t-test.md) | [F-Test](./f-test.md) →_

# Skewness and Kurtosis

Distribution shape characteristics that describe asymmetry and tail behavior of probability distributions.

## 🎯 Overview

Skewness and kurtosis are measures that describe the shape of a probability distribution beyond the basic measures of central tendency and spread.

## 📊 Skewness

### Definition

Skewness measures the asymmetry of a distribution around its mean.

### Types of Skewness

#### Negative Skewness (Left Skewness)

- **Mean < Median < Mode**
- **Longer or fatter tail on the left**
- **Outliers on the lower end of the distribution**
- **Most data concentrated on the right**

#### Positive Skewness (Right Skewness)

- **Mode < Median < Mean**
- **Longer or fatter tail on the right**
- **Outliers on the higher end of the distribution**
- **Most data concentrated on the left**

#### Zero Skewness

- **Symmetric distribution**
- **Mean = Median = Mode**
- **Equal tails on both sides**

## 📈 Kurtosis

### Definition

Kurtosis measures the "tailedness" of a distribution - how heavy or light the tails are compared to a normal distribution.

### Types of Kurtosis

#### Excess Kurtosis

- **Positive excess kurtosis**: More extreme events, both gains and losses
- **Negative excess kurtosis**: Fewer extreme events than normal distribution
- **Zero excess kurtosis**: Normal distribution

### Financial Implications

- **Extreme returns especially on the downside** (large negative returns)
- **Risk assessment** in financial modeling
- **Portfolio management** considerations

## 🔍 Mathematical Definitions

### Skewness Formula

```
Skewness = E[(X - μ)³] / σ³
```

### Kurtosis Formula

```
Kurtosis = E[(X - μ)⁴] / σ⁴
Excess Kurtosis = Kurtosis - 3
```

## 📊 Visual Interpretation

### Skewness

#### Negative Skewness (Left Skewed)

```
    Distribution Shape:

    Frequency
        ^
        |    /\
        |   /  \
        |  /    \_______
        | /             \
        |/               \
        +-------------------> Values
        |  Mode  Median  Mean
```

**Characteristics:**

- **Tail extends to the left** (negative values)
- **Most data concentrated on the right**
- **Mean < Median < Mode**
- **Outliers pull mean down**

#### Positive Skewness (Right Skewed)

```
    Distribution Shape:

    Frequency
        ^
        |       /\
        |      /  \
        |_____/    \
        |           \
        |            \
        +-------------------> Values
        |  Mean  Median  Mode
```

**Characteristics:**

- **Tail extends to the right** (positive values)
- **Most data concentrated on the left**
- **Mode < Median < Mean**
- **Outliers pull mean up**

#### Zero Skewness (Symmetric)

```
    Distribution Shape:

    Frequency
        ^
        |     /\
        |    /  \
        |   /    \
        |  /      \
        | /        \
        +-------------------> Values
        |  Mode=Median=Mean
```

**Characteristics:**

- **Perfectly symmetric around center**
- **Equal tails on both sides**
- **Mean = Median = Mode**
- **No directional bias**

### Kurtosis

#### High Kurtosis (Leptokurtic)

```
    Distribution Shape:

    Frequency
        ^
        |      /\
        |     /  \
        |    /    \
        |   /      \
        |  /        \___
        | /             \
        +-------------------> Values
        |  Sharp peak, fat tails
```

**Characteristics:**

- **Sharp, narrow peak**
- **Heavy, fat tails**
- **More extreme values than normal**
- **Concentrated around mean with outliers**

#### Low Kurtosis (Platykurtic)

```
    Distribution Shape:

    Frequency
        ^
        |   _______
        |  /       \
        | /         \
        |/           \
        |             \
        |              \
        +-------------------> Values
        |  Flat peak, thin tails
```

**Characteristics:**

- **Broad, flat peak**
- **Thin, light tails**
- **Fewer extreme values**
- **More uniform distribution**

#### Normal Kurtosis (Mesokurtic)

```
    Distribution Shape:

    Frequency
        ^
        |     /\
        |    /  \
        |   /    \
        |  /      \
        | /        \
        +-------------------> Values
        |  Standard normal shape
```

**Characteristics:**

- **Standard bell curve shape**
- **Moderate peak and tails**
- **Baseline for comparison**
- **Excess kurtosis = 0**

## 💼 Financial Applications

### Risk Management

- **Tail risk assessment**
- **Value at Risk (VaR) calculations**
- **Stress testing scenarios**

### Portfolio Analysis

- **Return distribution analysis**
- **Asset allocation decisions**
- **Risk-adjusted performance measures**

### Market Analysis

- **Volatility clustering**
- **Fat tail events**
- **Black swan event preparation**

## 🔗 Impact on Statistical Tests

### Assumptions

- **Many tests assume normality**
- **Skewed data may violate assumptions**
- **High kurtosis affects standard errors**

### Solutions

- **Transform data** (log, square root)
- **Use robust methods**
- **Non-parametric alternatives**

## 📝 Practical Examples

### Example 1: Income Distribution

- **Typically right-skewed**
- **Most people earn below mean**
- **Few high earners create long right tail**

### Example 2: Stock Returns

- **Often leptokurtic** (high kurtosis)
- **More extreme events than normal distribution**
- **Risk models need to account for fat tails**

### Example 3: Test Scores

- **May be left-skewed** (ceiling effect)
- **Most students score near maximum**
- **Few low scores create left tail**

## 🚨 Common Mistakes

1. **Ignoring skewness in hypothesis tests**
2. **Assuming normal distribution without checking**
3. **Not considering financial implications**
4. **Focusing only on mean and variance**

## 💡 Best Practices

1. **Always examine distribution shape**
2. **Consider transformations for skewed data**
3. **Use appropriate statistical methods**
4. **Account for kurtosis in risk models**
5. **Visualize distributions before analysis**

---

_← [Statistical Concepts](./README.md) | [Type I and Type II Errors](./type-errors.md) →_

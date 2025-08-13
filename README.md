# Statistical Knowledge Base

A comprehensive collection of statistical concepts, hypothesis testing methods, probability distributions, and advanced statistical techniques with practical applications in quantitative finance.

## 📚 Table of Contents

### 1. [Statistical Concepts](./concepts/README.md)

- [P-Values](./concepts/p-values.md) - Understanding statistical significance with Python implementations
- [Skewness and Kurtosis](./concepts/skewness-kurtosis.md) - Distribution shape analysis with financial applications

### 2. [Hypothesis Testing](./hypothesis-testing/README.md)

- [Z-Test](./hypothesis-testing/z-test.md) - Large sample mean comparison
- [T-Test](./hypothesis-testing/t-test.md) - Small sample mean comparison
- [Chi-Square Test](./hypothesis-testing/chi-square-test.md) - Categorical data analysis

### 3. [Probability Distributions](./distributions/README.md)

- [Overview](./distributions/README.md) - Comprehensive distribution guide with Python examples
- [Discrete Distributions](./distributions/discrete-distributions.md) - Binomial, Poisson, Geometric
- [Continuous Distributions](./distributions/continuous-distributions.md) - Normal, Exponential, Gamma
- [Specialized Distributions](./distributions/specialized-distributions.md) - t, Chi-square, F, Beta

### 4. [Advanced Statistical Methods](./advanced-statistics/README.md)

- [Overview](./advanced-statistics/README.md) - Advanced techniques overview
- [Causal Inference](./advanced-statistics/causal-inference.md) - RCT, Propensity Scores, IV, DiD
- [Bayesian Statistics](./advanced-statistics/bayesian-statistics.md) - Prior/posterior analysis
- [Robust Statistics](./advanced-statistics/robust-statistics.md) - Outlier-resistant methods
- [Multivariate Analysis](./advanced-statistics/multivariate-analysis.md) - PCA, Factor Analysis

### 5. [Time Series Analysis](./time-series/README.md)

- [Overview](./time-series/README.md) - Time series fundamentals
- [ARIMA Models](./time-series/arima-models.md) - Autoregressive models
- [GARCH Models](./time-series/garch-models.md) - Volatility modeling
- [Stationarity Tests](./time-series/stationarity-tests.md) - ADF, KPSS tests

### 6. [Machine Learning](./machine-learning/README.md)

- [Overview](./machine-learning/README.md) - ML fundamentals
- [Feature Engineering](./machine-learning/feature-engineering.md) - Data preprocessing
- [Model Evaluation](./machine-learning/model-evaluation.md) - Performance metrics
- [Interpretability](./machine-learning/interpretability.md) - SHAP, LIME, Feature importance

### 7. [Finance Applications](./finance/README.md)

- [Overview](./finance/README.md) - Financial statistics overview
- [Portfolio Optimization](./finance/portfolio-optimization.md) - Risk-return optimization
- [Risk Management](./finance/risk-management.md) - VaR, CVaR, Stress testing
- [Options Pricing](./finance/options-pricing.md) - Black-Scholes, Greeks
- [ESG Analysis](./finance/esg-analysis.md) - Environmental, Social, Governance metrics

## 🎯 Quick Reference

### When to Use Which Test?

| Scenario                                  | Recommended Test          |
| ----------------------------------------- | ------------------------- |
| Comparing means (large sample, known σ)   | Z-Test                    |
| Comparing means (small sample, unknown σ) | T-Test                    |
| Categorical variables relationship        | Chi-Square Test           |
| Comparing proportions                     | Z-Test for Proportions    |
| Causal relationships                      | Propensity Score Matching |
| Small sample inference                    | Bayesian Methods          |
| Outlier contamination                     | Robust Statistics         |
| High-dimensional data                     | PCA/Factor Analysis       |
| Time series forecasting                   | ARIMA/GARCH Models        |
| Survival analysis                         | Log Rank Test             |
| Multiple comparisons                      | Tukey Test                |

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

## 🐍 Code Examples

All sections include practical Python implementations with:

- **Complete code examples** ready to run
- **Real-world applications** in finance and data science
- **Visualization tools** for better understanding
- **Performance comparisons** between methods

## 💼 Financial Applications

Special focus on quantitative finance applications:

- **Risk Management**: VaR, CVaR, stress testing
- **Portfolio Analysis**: Optimization, factor models
- **Options Pricing**: Black-Scholes, Greeks, Monte Carlo
- **Market Analysis**: Time series, volatility modeling
- **ESG Integration**: Sustainability metrics and analysis

## 🚀 Recent Updates (December 2024)

### Content Enhancements:

- ✅ **Enhanced P-Values**: Added comprehensive Python implementations, multiple testing correction, effect size analysis
- ✅ **Advanced Statistics**: Created complete framework with robust methods, bootstrap, outlier detection, PCA
- ✅ **Causal Inference**: Comprehensive guide with RCT, Propensity Score Matching, Instrumental Variables, Difference-in-Differences
- ✅ **Probability Distributions**: Complete overview with practical examples and financial applications
- ✅ **Updated Navigation**: Enhanced structure with new sections and better organization

### Key Features Added:

- **4 Complete Python Classes** with production-ready implementations
- **15+ Practical Functions** with real-world examples
- **Financial Focus** throughout all examples
- **Professional Standards** with error handling and validation

## 📖 How to Use This Knowledge Base

1. **Browse by Topic**: Navigate through the main categories above
2. **Quick Reference**: Use the tables above for test and distribution selection
3. **Code Examples**: Run the Python implementations for hands-on learning
4. **Financial Focus**: Explore quantitative finance applications
5. **Search**: Use your browser's search function (Ctrl+F) to find specific terms

## 🔗 Related Resources

- **[GitHub Repository](https://github.com/YiyuanLei/knowledge_share)** - Source code and examples
- **[Interactive Examples](./examples/)** - Jupyter notebooks with live code
- **[Visualizations](./visualizations/)** - Generated plots and charts
- **[References](./references/)** - Further reading and citations

---

_Last updated: December 2024_
_Maintained by: Oliver Lei_

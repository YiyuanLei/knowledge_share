# Advanced Statistical Methods

Advanced statistical techniques for complex data analysis, causal inference, and modern statistical modeling.

## 🎯 Overview

Advanced statistical methods extend beyond basic hypothesis testing to address complex real-world problems including causal relationships, uncertainty quantification, and robust analysis.

## 📊 Method Categories

### Causal Inference

- **[Observational Studies](./causal-inference.md)** - Correlation vs causation
- **[Propensity Score Methods](./propensity-scores.md)** - Matching and weighting
- **[Instrumental Variables](./instrumental-variables.md)** - Natural experiments
- **[Difference-in-Differences](./difference-in-differences.md)** - Quasi-experimental design

### Bayesian Statistics

- **[Bayesian Inference](./bayesian-statistics.md)** - Prior and posterior analysis
- **[MCMC Methods](./mcmc-methods.md)** - Markov Chain Monte Carlo
- **[Bayesian Model Selection](./bayesian-model-selection.md)** - Model comparison
- **[Hierarchical Models](./hierarchical-models.md)** - Multi-level modeling

### Robust Statistics

- **[Outlier Detection](./outlier-detection.md)** - Identifying unusual observations
- **[Robust Estimation](./robust-estimation.md)** - Median, MAD, trimmed means
- **[Bootstrap Methods](./bootstrap-methods.md)** - Resampling techniques
- **[Permutation Tests](./permutation-tests.md)** - Distribution-free testing

### Multivariate Analysis

- **[Principal Component Analysis](./pca.md)** - Dimensionality reduction
- **[Factor Analysis](./factor-analysis.md)** - Latent variable modeling
- **[Canonical Correlation](./canonical-correlation.md)** - Multi-set relationships
- **[Multidimensional Scaling](./mds.md)** - Distance-based visualization

## 🔍 Quick Reference

### When to Use Which Method?

| Problem Type          | Recommended Method           |
| --------------------- | ---------------------------- |
| Causal relationships  | Propensity Score Matching    |
| Small sample sizes    | Bayesian Inference           |
| Outlier contamination | Robust Statistics            |
| High-dimensional data | PCA/Factor Analysis          |
| Non-normal data       | Bootstrap/Permutation        |
| Hierarchical data     | Mixed Effects Models         |
| Time-varying effects  | Survival Analysis            |
| Multiple outcomes     | MANOVA/Canonical Correlation |

## 🐍 Python Implementation

### Advanced Statistical Framework

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedStatistics:
    def __init__(self):
        self.results = {}

    def robust_regression(self, X, y, method='huber'):
        """
        Perform robust regression using different methods

        Parameters:
        X: Feature matrix
        y: Target variable
        method: 'huber', 'ransac', or 'theil_sen'

        Returns:
        model: Fitted robust regression model
        """
        from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

        models = {
            'huber': HuberRegressor(),
            'ransac': RANSACRegressor(),
            'theil_sen': TheilSenRegressor()
        }

        if method not in models:
            raise ValueError(f"Method {method} not supported")

        model = models[method]
        model.fit(X, y)

        return model

    def bootstrap_confidence_interval(self, data, statistic, n_bootstrap=10000, confidence=0.95):
        """
        Calculate bootstrap confidence interval

        Parameters:
        data: Array of observations
        statistic: Function to calculate statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

        Returns:
        ci_lower, ci_upper: Confidence interval bounds
        """
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic(bootstrap_sample))

        # Calculate confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1-alpha/2) * 100)

        return ci_lower, ci_upper

    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """
        Detect outliers using various methods

        Parameters:
        data: Array of observations
        method: 'iqr', 'zscore', or 'isolation_forest'
        threshold: Threshold for outlier detection

        Returns:
        outliers: Boolean array indicating outliers
        """
        if method == 'iqr':
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold

        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(data.reshape(-1, 1)) == -1

        else:
            raise ValueError(f"Method {method} not supported")

        return outliers

    def principal_component_analysis(self, X, n_components=None):
        """
        Perform Principal Component Analysis

        Parameters:
        X: Feature matrix
        n_components: Number of components to retain

        Returns:
        pca: Fitted PCA object
        explained_variance_ratio: Proportion of variance explained
        """
        pca = PCA(n_components=n_components)
        pca.fit(X)

        return pca, pca.explained_variance_ratio_

# Example usage
def advanced_statistics_example():
    """Demonstrate advanced statistical methods"""

    # Generate sample data with outliers
    np.random.seed(42)
    n_samples = 100

    # Normal data
    normal_data = np.random.normal(0, 1, n_samples)

    # Add outliers
    data_with_outliers = np.concatenate([normal_data, [5, -5, 8, -8]])

    # Initialize advanced statistics
    adv_stats = AdvancedStatistics()

    # Detect outliers
    outliers_iqr = adv_stats.detect_outliers(data_with_outliers, method='iqr')
    outliers_zscore = adv_stats.detect_outliers(data_with_outliers, method='zscore')

    print("Outlier Detection Results:")
    print(f"Data points: {len(data_with_outliers)}")
    print(f"Outliers (IQR method): {np.sum(outliers_iqr)}")
    print(f"Outliers (Z-score method): {np.sum(outliers_zscore)}")

    # Bootstrap confidence interval for mean
    def mean_statistic(data):
        return np.mean(data)

    ci_lower, ci_upper = adv_stats.bootstrap_confidence_interval(
        data_with_outliers, mean_statistic
    )

    print(f"\nBootstrap 95% CI for mean: [{ci_lower:.3f}, {ci_upper:.3f}]")

    # PCA example
    X = np.random.randn(100, 5)  # 5 features
    pca, explained_var = adv_stats.principal_component_analysis(X, n_components=2)

    print(f"\nPCA Results:")
    print(f"Explained variance ratio: {explained_var}")
    print(f"Total variance explained: {np.sum(explained_var):.3f}")

# Run example
advanced_statistics_example()
```

## 💼 Financial Applications

### Risk Management

- **Robust Statistics**: Outlier-resistant risk measures
- **Bootstrap Methods**: Confidence intervals for VaR
- **Bayesian Inference**: Uncertainty in risk parameters

### Portfolio Analysis

- **PCA**: Factor modeling and risk decomposition
- **Robust Regression**: Outlier-resistant asset pricing
- **Bootstrap**: Resampling for portfolio optimization

### Market Analysis

- **Causal Inference**: Impact of policy changes
- **Time Series**: Advanced forecasting methods
- **Multivariate**: Cross-asset relationships

## 🔗 Related Topics

- **[Basic Statistics](../concepts/README.md)** - Foundation concepts
- **[Hypothesis Testing](../hypothesis-testing/README.md)** - Traditional methods
- **[Time Series Analysis](../time-series/README.md)** - Temporal data
- **[Machine Learning](../machine-learning/README.md)** - Predictive modeling

## 📚 Further Reading

- **Causal Inference**: Morgan & Winship
- **Bayesian Data Analysis**: Gelman et al.
- **Robust Statistics**: Huber & Ronchetti
- **Multivariate Analysis**: Johnson & Wichern

---

_← [Statistical Concepts](../concepts/README.md) | [Machine Learning](../machine-learning/README.md) →_

# Causal Inference

Methods for establishing causal relationships from observational data, moving beyond correlation to understand true cause-and-effect relationships.

## 🎯 Overview

Causal inference aims to answer "what if" questions by estimating the causal effect of one variable on another, accounting for confounding factors and selection bias.

## 🔍 Key Concepts

### Correlation vs Causation

**Correlation**: Statistical relationship between variables
**Causation**: One variable directly influences another

### Confounding Variables

Variables that influence both the treatment and outcome, creating spurious associations.

### Selection Bias

Systematic differences between treated and control groups that affect outcomes.

## 📊 Causal Inference Framework

### Potential Outcomes Framework

For each individual, we define:

- **Y(1)**: Outcome if treated
- **Y(0)**: Outcome if not treated
- **Treatment Effect**: Y(1) - Y(0)

### Average Treatment Effect (ATE)

```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

### Fundamental Problem of Causal Inference

We can never observe both Y(1) and Y(0) for the same individual.

## 🛠️ Methods

### 1. Randomized Controlled Trials (RCT)

**Gold Standard**: Random assignment eliminates confounding.

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def simulate_rct(n_treatment=100, n_control=100, true_effect=0.5):
    """
    Simulate a randomized controlled trial

    Parameters:
    n_treatment: Number of treated individuals
    n_control: Number of control individuals
    true_effect: True treatment effect

    Returns:
    results: Dictionary with trial results
    """
    np.random.seed(42)

    # Generate potential outcomes
    baseline_outcome = np.random.normal(10, 2, n_treatment + n_control)

    # Treatment effect
    treatment_outcome = baseline_outcome + true_effect

    # Assign treatment randomly
    treatment_assignment = np.random.choice(
        [0, 1], size=n_treatment + n_control, replace=False
    )

    # Observed outcomes
    observed_outcome = np.where(treatment_assignment == 1,
                               treatment_outcome, baseline_outcome)

    # Create results dataframe
    results_df = pd.DataFrame({
        'treatment': treatment_assignment,
        'outcome': observed_outcome,
        'baseline': baseline_outcome
    })

    # Calculate treatment effect
    treatment_mean = results_df[results_df['treatment'] == 1]['outcome'].mean()
    control_mean = results_df[results_df['treatment'] == 0]['outcome'].mean()
    estimated_effect = treatment_mean - control_mean

    # Statistical test
    treatment_group = results_df[results_df['treatment'] == 1]['outcome']
    control_group = results_df[results_df['treatment'] == 0]['outcome']
    t_stat, p_value = stats.ttest_ind(treatment_group, control_group)

    return {
        'estimated_effect': estimated_effect,
        'true_effect': true_effect,
        'p_value': p_value,
        'treatment_mean': treatment_mean,
        'control_mean': control_mean,
        'data': results_df
    }

# Example usage
rct_results = simulate_rct(n_treatment=100, n_control=100, true_effect=0.5)
print(f"Estimated treatment effect: {rct_results['estimated_effect']:.3f}")
print(f"True treatment effect: {rct_results['true_effect']:.3f}")
print(f"P-value: {rct_results['p_value']:.4f}")
```

### 2. Propensity Score Methods

**Matching**: Pair treated and control individuals with similar characteristics.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

class PropensityScoreMatching:
    def __init__(self):
        self.propensity_model = None
        self.propensity_scores = None

    def estimate_propensity_scores(self, X, treatment):
        """
        Estimate propensity scores using logistic regression

        Parameters:
        X: Covariate matrix
        treatment: Treatment assignment (0/1)

        Returns:
        propensity_scores: Estimated propensity scores
        """
        self.propensity_model = LogisticRegression(random_state=42)
        self.propensity_model.fit(X, treatment)
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]

        return self.propensity_scores

    def match_individuals(self, treatment, propensity_scores, method='nearest'):
        """
        Match treated and control individuals

        Parameters:
        treatment: Treatment assignment
        propensity_scores: Estimated propensity scores
        method: Matching method ('nearest', 'caliper')

        Returns:
        matched_pairs: Indices of matched pairs
        """
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        treated_scores = propensity_scores[treated_idx]
        control_scores = propensity_scores[control_idx]

        if method == 'nearest':
            # Nearest neighbor matching
            nbrs = NearestNeighbors(n_neighbors=1)
            nbrs.fit(control_scores.reshape(-1, 1))
            distances, indices = nbrs.kneighbors(treated_scores.reshape(-1, 1))

            matched_pairs = list(zip(treated_idx, control_idx[indices.flatten()]))

        elif method == 'caliper':
            # Caliper matching
            caliper = 0.1 * np.std(propensity_scores)
            matched_pairs = []

            for i, treated_score in enumerate(treated_scores):
                # Find controls within caliper
                within_caliper = np.abs(control_scores - treated_score) <= caliper
                if np.any(within_caliper):
                    # Choose closest match
                    distances = np.abs(control_scores - treated_score)
                    closest_idx = np.argmin(distances[within_caliper])
                    control_matches = control_idx[within_caliper]
                    matched_pairs.append((treated_idx[i], control_matches[closest_idx]))

        return matched_pairs

    def estimate_treatment_effect(self, outcome, matched_pairs):
        """
        Estimate treatment effect using matched pairs

        Parameters:
        outcome: Outcome variable
        matched_pairs: Indices of matched pairs

        Returns:
        treatment_effect: Estimated average treatment effect
        """
        pair_differences = []

        for treated_idx, control_idx in matched_pairs:
            diff = outcome[treated_idx] - outcome[control_idx]
            pair_differences.append(diff)

        treatment_effect = np.mean(pair_differences)

        return treatment_effect

# Example usage
def propensity_score_example():
    """Demonstrate propensity score matching"""
    np.random.seed(42)
    n_samples = 1000

    # Generate covariates
    age = np.random.normal(45, 15, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    education = np.random.normal(12, 3, n_samples)

    # Generate propensity score (probability of treatment)
    propensity = 1 / (1 + np.exp(-(-2 + 0.05*age + 0.00001*income + 0.1*education)))

    # Assign treatment based on propensity score
    treatment = np.random.binomial(1, propensity)

    # Generate outcomes with treatment effect
    baseline_outcome = 50 + 0.5*age + 0.001*income + 2*education + np.random.normal(0, 5, n_samples)
    treatment_effect = 10  # True treatment effect
    outcome = baseline_outcome + treatment_effect * treatment

    # Create feature matrix
    X = np.column_stack([age, income, education])

    # Apply propensity score matching
    psm = PropensityScoreMatching()
    propensity_scores = psm.estimate_propensity_scores(X, treatment)
    matched_pairs = psm.match_individuals(treatment, propensity_scores, method='nearest')
    estimated_effect = psm.estimate_treatment_effect(outcome, matched_pairs)

    print(f"Number of matched pairs: {len(matched_pairs)}")
    print(f"Estimated treatment effect: {estimated_effect:.3f}")
    print(f"True treatment effect: {treatment_effect:.3f}")

    return psm, matched_pairs, estimated_effect

# Run example
psm_results = propensity_score_example()
```

### 3. Instrumental Variables

**Natural Experiments**: Use external variation to identify causal effects.

```python
class InstrumentalVariables:
    def __init__(self):
        self.first_stage_model = None
        self.second_stage_model = None

    def two_stage_least_squares(self, X, Z, y):
        """
        Perform two-stage least squares estimation

        Parameters:
        X: Endogenous variable
        Z: Instrumental variable
        y: Outcome variable

        Returns:
        iv_estimate: Instrumental variables estimate
        """
        from sklearn.linear_model import LinearRegression

        # First stage: Regress X on Z
        self.first_stage_model = LinearRegression()
        self.first_stage_model.fit(Z.reshape(-1, 1), X)
        X_hat = self.first_stage_model.predict(Z.reshape(-1, 1))

        # Second stage: Regress y on predicted X
        self.second_stage_model = LinearRegression()
        self.second_stage_model.fit(X_hat.reshape(-1, 1), y)

        iv_estimate = self.second_stage_model.coef_[0]

        return iv_estimate

    def check_instrument_validity(self, Z, X, y):
        """
        Check instrumental variable assumptions

        Parameters:
        Z: Instrumental variable
        X: Endogenous variable
        y: Outcome variable

        Returns:
        validity_checks: Dictionary with validity measures
        """
        # Relevance: Z should be correlated with X
        relevance_corr = np.corrcoef(Z, X)[0, 1]

        # Exclusion: Z should not directly affect y (test with residuals)
        from sklearn.linear_model import LinearRegression

        # Regress y on X
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y)
        residuals = y - model.predict(X.reshape(-1, 1))

        # Check correlation between Z and residuals
        exclusion_corr = np.corrcoef(Z, residuals)[0, 1]

        return {
            'relevance_correlation': relevance_corr,
            'exclusion_correlation': exclusion_corr,
            'instrument_strength': abs(relevance_corr) > 0.1
        }

# Example: Distance to college as instrument for education
def instrumental_variables_example():
    """Demonstrate instrumental variables with distance to college"""
    np.random.seed(42)
    n_samples = 1000

    # Generate instrumental variable (distance to college)
    distance = np.random.exponential(50, n_samples)  # Distance in miles

    # Generate unobserved ability (confounder)
    ability = np.random.normal(0, 1, n_samples)

    # Education depends on distance and ability
    education = 12 + 0.5*ability - 0.01*distance + np.random.normal(0, 1, n_samples)

    # Income depends on education and ability
    income = 30000 + 2000*education + 5000*ability + np.random.normal(0, 5000, n_samples)

    # Apply instrumental variables
    iv = InstrumentalVariables()
    iv_estimate = iv.two_stage_least_squares(education, distance, income)
    validity = iv.check_instrument_validity(distance, education, income)

    # Compare with OLS (biased due to omitted variable)
    from sklearn.linear_model import LinearRegression
    ols_model = LinearRegression()
    ols_model.fit(education.reshape(-1, 1), income)
    ols_estimate = ols_model.coef_[0]

    print("Instrumental Variables Results:")
    print(f"IV estimate: {iv_estimate:.2f}")
    print(f"OLS estimate: {ols_estimate:.2f}")
    print(f"True effect: 2000.00")
    print(f"Relevance correlation: {validity['relevance_correlation']:.3f}")
    print(f"Exclusion correlation: {validity['exclusion_correlation']:.3f}")

    return iv_estimate, ols_estimate, validity

# Run example
iv_results = instrumental_variables_example()
```

### 4. Difference-in-Differences

**Before-After Comparison**: Compare changes over time between treated and control groups.

```python
class DifferenceInDifferences:
    def __init__(self):
        self.results = {}

    def estimate_did(self, data):
        """
        Estimate difference-in-differences effect

        Parameters:
        data: DataFrame with columns [group, time, outcome]

        Returns:
        did_estimate: Difference-in-differences estimate
        """
        # Calculate group-time means
        means = data.groupby(['group', 'time'])['outcome'].mean().unstack()

        # Calculate differences
        control_diff = means.loc[0, 1] - means.loc[0, 0]  # Control: after - before
        treatment_diff = means.loc[1, 1] - means.loc[1, 0]  # Treatment: after - before

        # DID estimate
        did_estimate = treatment_diff - control_diff

        self.results = {
            'control_before': means.loc[0, 0],
            'control_after': means.loc[0, 1],
            'treatment_before': means.loc[1, 0],
            'treatment_after': means.loc[1, 1],
            'control_diff': control_diff,
            'treatment_diff': treatment_diff,
            'did_estimate': did_estimate
        }

        return did_estimate

    def plot_did(self, data):
        """Plot difference-in-differences results"""
        means = data.groupby(['group', 'time'])['outcome'].mean().unstack()

        plt.figure(figsize=(10, 6))

        # Plot lines
        plt.plot([0, 1], [means.loc[0, 0], means.loc[0, 1]],
                'b-o', label='Control', linewidth=2, markersize=8)
        plt.plot([0, 1], [means.loc[1, 0], means.loc[1, 1]],
                'r-s', label='Treatment', linewidth=2, markersize=8)

        # Add parallel trends assumption
        plt.plot([0, 1], [means.loc[1, 0], means.loc[1, 0] + means.loc[0, 1] - means.loc[0, 0]],
                'r--', alpha=0.5, label='Counterfactual (parallel trends)')

        plt.xlabel('Time (0=Before, 1=After)')
        plt.ylabel('Outcome')
        plt.title('Difference-in-Differences Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example: Policy intervention
def difference_in_differences_example():
    """Demonstrate difference-in-differences with policy intervention"""
    np.random.seed(42)
    n_samples = 2000

    # Generate data
    group = np.random.choice([0, 1], size=n_samples)  # 0=control, 1=treatment
    time = np.random.choice([0, 1], size=n_samples)   # 0=before, 1=after

    # Baseline outcome
    baseline = 100 + 10*group + np.random.normal(0, 10, n_samples)

    # Time trend (same for both groups)
    time_trend = 5*time

    # Treatment effect (only for treatment group after intervention)
    treatment_effect = 15 * group * time

    # Final outcome
    outcome = baseline + time_trend + treatment_effect

    # Create dataframe
    data = pd.DataFrame({
        'group': group,
        'time': time,
        'outcome': outcome
    })

    # Apply DID
    did = DifferenceInDifferences()
    did_estimate = did.estimate_did(data)

    print("Difference-in-Differences Results:")
    print(f"Control before: {did.results['control_before']:.2f}")
    print(f"Control after: {did.results['control_after']:.2f}")
    print(f"Treatment before: {did.results['treatment_before']:.2f}")
    print(f"Treatment after: {did.results['treatment_after']:.2f}")
    print(f"Control difference: {did.results['control_diff']:.2f}")
    print(f"Treatment difference: {did.results['treatment_diff']:.2f}")
    print(f"DID estimate: {did_estimate:.2f}")
    print(f"True treatment effect: 15.00")

    # Plot results
    did.plot_did(data)

    return did_estimate

# Run example
did_results = difference_in_differences_example()
```

## 💼 Financial Applications

### Market Impact Analysis

- **Event Studies**: Impact of announcements on stock prices
- **Policy Changes**: Effect of regulatory changes on market behavior
- **Trading Strategies**: Causal impact of algorithmic trading

### Credit Risk

- **Loan Programs**: Effect of lending policies on default rates
- **Credit Scoring**: Impact of score changes on borrowing behavior

### Portfolio Management

- **Factor Models**: Causal relationships between factors and returns
- **Risk Management**: Impact of risk controls on portfolio performance

## 🔗 Related Topics

- **[Propensity Score Methods](./propensity-scores.md)** - Matching and weighting
- **[Bayesian Statistics](./bayesian-statistics.md)** - Uncertainty quantification
- **[Time Series Analysis](../time-series/README.md)** - Temporal causal effects
- **[Machine Learning](../machine-learning/README.md)** - Predictive modeling

## 📚 Further Reading

- **Causal Inference**: Morgan & Winship
- **Mostly Harmless Econometrics**: Angrist & Pischke
- **Causal Inference in Statistics**: Pearl et al.
- **Counterfactuals and Causal Inference**: Morgan & Winship

---

_← [Advanced Statistics](./README.md) | [Propensity Score Methods](./propensity-scores.md) →_

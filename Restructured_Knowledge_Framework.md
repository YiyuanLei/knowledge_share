# Knowledge Framework: AI, Statistics, and Quantitative Methods

## Table of Contents

1. [AI and Machine Learning](#ai-and-machine-learning)
2. [Statistical Methods](#statistical-methods)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Financial Concepts](#financial-concepts)
5. [Time Series Analysis](#time-series-analysis)
6. [Optimization and Algorithms](#optimization-and-algorithms)
7. [Communication and Evaluation](#communication-and-evaluation)
8. [Web Development](#web-development)

---

## AI and Machine Learning

### Deployment Strategies

When deploying AI systems, understanding the trade-offs between latency and throughput is crucial for choosing the right approach:

**Batch Processing**

- **Characteristics**: High throughput, high latency (hours to days)
- **Use Case**: Processing large datasets where immediate results aren't needed
- **Example**: Analyzing quarterly financial reports to generate investment insights
- **Considerations**: Cost-effective for large-scale operations, but not suitable for real-time applications

**Streaming Processing**

- **Characteristics**: Moderate throughput, moderate latency (seconds to minutes)
- **Use Case**: Processing data as it arrives in micro-batches
- **Example**: Personalizing marketing messages based on user behavior patterns
- **Considerations**: Balances efficiency with responsiveness

**Real-time Processing**

- **Characteristics**: Variable throughput, low latency (milliseconds)
- **Use Case**: Immediate response requirements
- **Example**: Chatbots, real-time fraud detection, live trading systems
- **Considerations**: Requires robust infrastructure and careful optimization

**Edge/Embedded Processing**

- **Characteristics**: Low throughput, low latency, device-dependent
- **Use Case**: Local processing on devices with limited resources
- **Example**: Voice-controlled smart home devices, autonomous vehicle sensors
- **Considerations**: Limited by device processing power and battery life

### Hardware Considerations

**CPU vs GPU Architecture**

_Central Processing Units (CPUs)_

- Sequential processing with few powerful cores
- Optimized for complex, single-threaded tasks
- General-purpose computing

_Graphics Processing Units (GPUs)_

- Parallel processing with thousands of smaller cores
- Optimized for repetitive, similar operations
- Specialized for specific workloads

**GPU Applications**

_Graphics and Visualization_

- Virtual desktop infrastructure
- Movie rendering and animation
- Gaming graphics processing
- Real-time visualization

_Artificial Intelligence_

- Machine learning model training
- Deep learning inference
- Neural network operations
- Pattern recognition tasks

_High-Performance Computing_

- Scientific simulations
- Financial modeling
- Cloud computing services
- Data center operations

**Cloud GPU Solutions**

- On-demand access to high-performance computing
- Cost-effective scaling for variable workloads
- Bare metal and virtualized options
- Managed services for simplified deployment

### Machine Learning Fundamentals

**Model Evaluation and Validation**

_Residual Analysis_

- Calculate residuals: Y_predicted - Y_actual
- Check for patterns indicating model issues
- Verify assumptions: zero mean, constant variance, normality
- Consider transformations (e.g., log transformation) for skewed data

_Overfitting Prevention_

- **Problem**: Small in-sample error, large out-of-sample error
- **Solutions**:
  1. **Increase Bias**:
     - Regularization (L1/Lasso, L2/Ridge)
     - Feature selection (forward, backward, stepwise)
     - Dimensionality reduction (PCA)
  2. **Reduce Variance**:
     - Cross-validation
     - Ensemble methods
     - Early stopping

**Ensemble Methods**

_Bagging (Bootstrap Aggregating)_

- Reduces variance by averaging multiple models
- Examples: Random Forest
- Increases model stability
- Less prone to overfitting

_Boosting_

- Reduces bias by combining weak learners
- Examples: AdaBoost, Gradient Boosting
- Can increase variance if not properly controlled
- Requires careful tuning and early stopping

**Common Pitfalls and Solutions**

_Data Quality Issues_

- **Exploratory Data Analysis**: Essential for understanding data distributions, correlations, and patterns
- **Data Leakage**: Using future information during training
  - Solution: Strict temporal separation of training and test data
  - Perform all preprocessing within cross-validation folds
- **Missing Values**: Handle systematically with appropriate imputation methods
- **Outliers**: Use robust methods (IQR, z-scores) and consider their impact on model choice

_Model Validation_

- **Multiple Test Sets**: Reserve several test sets for different evaluation phases
- **Avoid Test Set Overuse**: Each test set examination increases leakage risk
- **A/B Testing**: Implement proper experimental design for production validation
- **Baseline Models**: Always compare against simple, interpretable baselines

### Advanced Machine Learning Concepts

**Feature Engineering Insights**

_Time-Based Features_

- **Cyclical Encoding**: Convert time features (hour, day, month) to cyclical representations using sin/cos transformations
- **Lag Features**: Create features from past observations (t-1, t-7, t-30) for time series problems
- **Rolling Statistics**: Moving averages, standard deviations, and other window-based aggregations
- **Domain-Specific Features**: Business cycles, seasonal patterns, event indicators

_Interaction Features_

- **Polynomial Features**: Capture non-linear relationships (x², x³, x₁x₂)
- **Cross-Product Features**: Combine categorical variables meaningfully
- **Ratio Features**: Create meaningful ratios (e.g., debt-to-income, price-to-earnings)
- **Binning and Discretization**: Convert continuous variables to categorical for non-linear patterns

**Model Interpretability Techniques**

_Local Interpretability_

- **SHAP Values**: Explain individual predictions by measuring feature contributions
- **LIME**: Approximate complex models with interpretable local models
- **Partial Dependence Plots**: Show relationship between features and predictions
- **Individual Conditional Expectation**: Plot how predictions change with feature values

_Global Interpretability_

- **Feature Importance**: Rank features by their overall contribution to model performance
- **Permutation Importance**: Measure importance by randomizing feature values
- **Model-Agnostic Methods**: Techniques that work across different model types
- **Surrogate Models**: Train simple, interpretable models to approximate complex ones

**Production Deployment Considerations**

_Model Monitoring_

- **Data Drift Detection**: Monitor changes in feature distributions over time
- **Prediction Drift**: Track changes in model output distributions
- **Performance Degradation**: Alert when model accuracy drops below thresholds
- **A/B Testing Framework**: Systematic comparison of model versions

_Infrastructure Requirements_

- **Scalability**: Design for handling increased load and data volume
- **Latency Requirements**: Ensure predictions meet real-time constraints
- **Reliability**: Implement redundancy and failover mechanisms
- **Security**: Protect model inputs, outputs, and intellectual property

**Real-World Application Patterns**

_Recommendation Systems_

- **Collaborative Filtering**: User-item interaction matrices
- **Content-Based Filtering**: Item feature similarity
- **Hybrid Approaches**: Combine multiple recommendation strategies
- **Evaluation Metrics**: Precision@k, Recall@k, NDCG, MAP

_Fraud Detection_

- **Anomaly Detection**: Identify unusual patterns in transactions
- **Behavioral Analysis**: Track user behavior changes over time
- **Network Analysis**: Detect coordinated fraudulent activities
- **Real-Time Scoring**: Immediate risk assessment for transactions

_Customer Segmentation_

- **Clustering Algorithms**: K-means, hierarchical, DBSCAN
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for visualization
- **Business Validation**: Ensure segments are actionable and meaningful
- **Dynamic Segmentation**: Update segments as customer behavior evolves

---

## Statistical Methods

### Probability Distributions

**Discrete Distributions**

_Binomial Distribution_

- Models number of successes in n independent trials
- Parameters: n (trials), p (success probability)
- Mean: np, Variance: np(1-p)
- Applications: Quality control, survey responses, coin flips

_Poisson Distribution_

- Models rare events in fixed time/space intervals
- Parameter: λ (average rate)
- Mean = Variance = λ
- Applications: Customer arrivals, defect counts, accident rates

_Geometric Distribution_

- Models trials until first success
- Parameter: p (success probability)
- Memoryless property
- Applications: Equipment failure analysis, game theory

_Negative Binomial Distribution_

- Models trials until r successes
- Parameters: r (required successes), p (success probability)
- Generalizes geometric distribution
- Applications: Reliability testing, sports statistics

**Continuous Distributions**

_Normal Distribution_

- Bell-shaped, symmetric distribution
- Parameters: μ (mean), σ (standard deviation)
- Central Limit Theorem applications
- Foundation for many statistical tests

_Exponential Distribution_

- Models time between events in Poisson process
- Parameter: λ (rate parameter)
- Memoryless property
- Applications: Equipment lifetime, inter-arrival times

_Gamma Distribution_

- Generalizes exponential distribution
- Parameters: α (shape), β (rate)
- Flexible for modeling various shapes
- Applications: Insurance claims, rainfall modeling

_Weibull Distribution_

- Models failure rates with time-varying hazard
- Parameters: α (shape), β (scale)
- Generalizes exponential distribution
- Applications: Reliability analysis, wind speed modeling

### Distribution Characteristics

**Skewness and Kurtosis**

_Negative Skewness (Left-Skewed)_

- Mean < Median < Mode
- Longer tail on the left
- Outliers concentrated on lower end
- Examples: Income distributions, exam scores

_Excess Kurtosis_

- Measures tail heaviness relative to normal distribution
- Positive excess kurtosis: more extreme events
- Important for risk assessment in finance
- Affects statistical inference validity

### Hypothesis Testing

**P-Values and Significance**

- Probability of observing data as extreme as observed, given null hypothesis
- Not probability that null hypothesis is true
- Should be interpreted with effect size and practical significance
- Multiple testing corrections needed for multiple comparisons

**Common Tests**

- **t-test**: Comparing means (paired or independent)
- **Chi-square test**: Categorical data analysis
- **ANOVA**: Multiple group comparisons
- **Non-parametric tests**: When assumptions aren't met

### Advanced Statistical Methods

**Causal Inference Framework**

_Observational Studies_

- **Confounding Variables**: Factors that influence both treatment and outcome
- **Selection Bias**: Systematic differences between groups being compared
- **Measurement Error**: Inaccuracies in variable measurement affecting relationships
- **Reverse Causality**: When outcome affects treatment rather than vice versa

_Propensity Score Methods_

- **Propensity Score Matching**: Match treated and control units with similar propensity scores
- **Inverse Probability Weighting**: Weight observations by inverse of treatment probability
- **Stratification**: Divide data into strata based on propensity scores
- **Doubly Robust Estimation**: Combine outcome modeling with propensity score methods

**Bayesian Statistics**

_Prior Specification_

- **Informative Priors**: Use domain knowledge to specify prior distributions
- **Non-informative Priors**: Minimize prior influence on posterior results
- **Conjugate Priors**: Mathematical convenience for analytical solutions
- **Hierarchical Priors**: Model parameters as coming from common distributions

_Bayesian Inference_

- **Posterior Distributions**: Updated beliefs after observing data
- **Credible Intervals**: Bayesian equivalent of confidence intervals
- **Bayes Factors**: Compare evidence for competing hypotheses
- **Markov Chain Monte Carlo**: Numerical methods for posterior sampling

**Robust Statistics**

_Outlier-Resistant Methods_

- **Median and MAD**: Robust alternatives to mean and standard deviation
- **Trimmed Means**: Remove extreme values before averaging
- **Winsorization**: Replace extreme values with less extreme ones
- **Robust Regression**: Methods less sensitive to outliers (e.g., Huber, RANSAC)

_Distribution-Free Methods_

- **Rank-Based Tests**: Use data ranks instead of actual values
- **Bootstrap Methods**: Resample data to estimate sampling distributions
- **Permutation Tests**: Randomly shuffle data to test hypotheses
- **Non-parametric Confidence Intervals**: Bootstrap-based interval estimation

**Multivariate Analysis**

_Principal Component Analysis_

- **Dimensionality Reduction**: Reduce variables while preserving variance
- **Feature Extraction**: Create new uncorrelated features
- **Data Visualization**: Project high-dimensional data to 2D/3D
- **Noise Reduction**: Remove components with low variance

_Factor Analysis_

- **Latent Variable Modeling**: Identify underlying factors from observed variables
- **Factor Rotation**: Improve interpretability of factor structure
- **Confirmatory Factor Analysis**: Test hypothesized factor structure
- **Exploratory Factor Analysis**: Discover underlying factor structure

**Experimental Design**

_Randomized Controlled Trials_

- **Randomization**: Eliminate systematic differences between groups
- **Blinding**: Prevent bias from knowledge of treatment assignment
- **Blocking**: Control for known sources of variation
- **Factorial Designs**: Study multiple factors simultaneously

_Quasi-Experimental Designs_

- **Regression Discontinuity**: Natural cutoff points create treatment groups
- **Difference-in-Differences**: Compare changes over time between groups
- **Instrumental Variables**: Use external factors to identify causal effects
- **Natural Experiments**: Exploit naturally occurring treatment assignments

**Statistical Power and Sample Size**

_Power Analysis_

- **Effect Size**: Magnitude of difference or relationship to detect
- **Significance Level**: Probability of Type I error (α)
- **Power**: Probability of detecting true effect (1 - β)
- **Sample Size Planning**: Determine required sample size for desired power

_Practical Considerations_

- **Multiple Testing**: Adjust significance levels for multiple comparisons
- **Effect Size Interpretation**: Consider practical significance beyond statistical significance
- **Power Curves**: Visualize power across different effect sizes
- **Sequential Analysis**: Monitor results and stop when sufficient evidence is obtained

---

## Practical Applications and Case Studies

### Real-World Problem Solving Framework

**Problem Definition Phase**

- **Stakeholder Alignment**: Ensure all parties understand the problem and success criteria
- **Scope Definition**: Clearly define what's in and out of scope
- **Success Metrics**: Establish measurable outcomes and KPIs
- **Timeline and Resources**: Set realistic expectations for delivery

**Data Assessment Phase**

- **Data Availability**: Inventory existing data sources and quality
- **Gap Analysis**: Identify missing data needed for solution
- **Data Governance**: Ensure compliance with privacy and security requirements
- **Infrastructure Requirements**: Assess computational and storage needs

**Solution Development Phase**

- **Prototype Development**: Build minimum viable solution for validation
- **Iterative Refinement**: Incorporate feedback and improve model performance
- **Validation Testing**: Ensure solution works across different scenarios
- **Documentation**: Create comprehensive documentation for deployment and maintenance

### Industry-Specific Applications

**Financial Services**

_Credit Risk Modeling_

- **Application Scoring**: Predict likelihood of loan default
- **Behavioral Scoring**: Monitor ongoing credit risk based on payment patterns
- **Portfolio Risk Management**: Aggregate risk across multiple loans
- **Regulatory Compliance**: Ensure models meet regulatory requirements (Basel III, etc.)

_Algorithmic Trading_

- **Signal Generation**: Identify trading opportunities from market data
- **Risk Management**: Control position sizes and exposure limits
- **Execution Optimization**: Minimize market impact and transaction costs
- **Backtesting Framework**: Validate strategies on historical data

**Healthcare**

_Predictive Analytics_

- **Readmission Risk**: Predict likelihood of hospital readmission
- **Disease Progression**: Model disease development over time
- **Treatment Response**: Predict patient response to different treatments
- **Resource Planning**: Optimize staffing and resource allocation

_Medical Imaging_

- **Image Classification**: Automate diagnosis from medical images
- **Segmentation**: Identify specific regions in medical scans
- **Quality Assessment**: Ensure image quality meets diagnostic standards
- **Integration**: Connect AI systems with existing medical workflows

**E-commerce and Retail**

_Customer Lifetime Value_

- **Purchase Prediction**: Forecast future customer purchases
- **Churn Prevention**: Identify customers at risk of leaving
- **Personalization**: Tailor recommendations and marketing
- **Pricing Optimization**: Set optimal prices for maximum revenue

_Inventory Management_

- **Demand Forecasting**: Predict product demand across locations
- **Supply Chain Optimization**: Minimize costs while meeting demand
- **Seasonal Planning**: Account for seasonal variations in demand
- **New Product Introduction**: Forecast demand for new products

**Manufacturing and Operations**

_Predictive Maintenance_

- **Equipment Failure Prediction**: Forecast when equipment will fail
- **Maintenance Scheduling**: Optimize maintenance timing and resources
- **Quality Control**: Detect defects early in production process
- **Energy Optimization**: Reduce energy consumption while maintaining output

_Supply Chain Analytics_

- **Supplier Performance**: Evaluate and predict supplier reliability
- **Logistics Optimization**: Minimize transportation costs and time
- **Risk Assessment**: Identify and mitigate supply chain risks
- **Sustainability Metrics**: Track environmental impact of operations

### Cross-Industry Patterns

**Data Quality Challenges**

- **Missing Data**: Systematic patterns in missing values across industries
- **Data Consistency**: Variations in data collection and storage methods
- **Temporal Alignment**: Synchronizing data from different time zones and systems
- **Scale Differences**: Handling data at different granularities and frequencies

**Model Deployment Patterns**

- **Batch vs Real-time**: Choosing appropriate processing approach
- **Model Versioning**: Managing multiple model versions in production
- **A/B Testing**: Systematic comparison of model performance
- **Monitoring and Alerting**: Detecting model degradation and data drift

**Ethical Considerations**

- **Bias Detection**: Identifying and mitigating algorithmic bias
- **Fairness Metrics**: Ensuring equitable treatment across demographic groups
- **Transparency**: Making model decisions interpretable and explainable
- **Privacy Protection**: Safeguarding sensitive information in models

---

## Mathematical Foundations

### Calculus

**Key Concepts**

- **Derivatives**: Rate of change, optimization, sensitivity analysis
- **Integrals**: Area under curves, cumulative distributions, expected values
- **Taylor Series**: Function approximation, numerical methods
- **Multivariate Calculus**: Partial derivatives, gradients, optimization

**Applications in Data Science**

- Gradient descent optimization
- Maximum likelihood estimation
- Sensitivity analysis of models
- Numerical integration for complex distributions

### Linear Algebra

**Eigenvalue Decomposition**

- Diagonalization of square matrices
- Principal Component Analysis foundation
- Dimensionality reduction
- System stability analysis

**Matrix Operations**

- Matrix multiplication for transformations
- Inverses and pseudoinverses
- Singular Value Decomposition (SVD)
- Applications in recommendation systems

### Optimization

**Constrained Optimization**

- **KKT Conditions**: Necessary conditions for optimality
- **Lagrange Multipliers**: Handling equality constraints
- **Penalty Methods**: Converting constrained to unconstrained problems
- **Mixed Integer Programming**: Discrete and continuous variables

**Applications**

- Portfolio optimization
- Resource allocation
- Machine learning hyperparameter tuning
- Operations research problems

---

## Financial Concepts

### Market Structure

**Brokerage Firms**

- Intermediaries connecting buyers and sellers
- Facilitate trading in various securities
- Provide research, advice, and execution services
- Regulated entities with fiduciary responsibilities

**Securities Markets**

- **Publicly Traded Securities**: Freely traded on exchanges
  - Stocks: Ownership shares in companies
  - Bonds: Debt instruments with fixed payments
  - Options: Derivatives with contingent payoffs
  - Mutual Funds: Pooled investment vehicles

**Investment Vehicles**

_Hedge Funds_

- Private investment partnerships
- Professional management with performance fees
- Diverse strategies including leverage and derivatives
- Higher risk and return potential than traditional investments

### Quantitative Finance

**Risk Management**

- Value at Risk (VaR) calculations
- Stress testing and scenario analysis
- Portfolio diversification principles
- Correlation and covariance analysis

**Pricing Models**

- Black-Scholes option pricing
- Binomial tree models
- Monte Carlo simulations
- Risk-neutral valuation

---

## Time Series Analysis

### Fundamental Concepts

**Stationarity**

- Statistical properties constant over time
- Required for many time series methods
- Testing: ADF, KPSS, visual inspection
- Achieving: differencing, transformations

**Autocorrelation**

- Correlation between observations at different lags
- Identifying patterns and dependencies
- Partial autocorrelation for order determination
- Applications in forecasting and modeling

### ARIMA Models

**Components**

- **AR (Autoregressive)**: Current value depends on past values
- **MA (Moving Average)**: Current value depends on past errors
- **I (Integration)**: Differencing to achieve stationarity
- **Seasonal Components**: Periodic patterns

**Model Selection**

- AIC/BIC for model comparison
- Residual analysis for adequacy
- Cross-validation for forecasting accuracy
- Box-Jenkins methodology

### Advanced Models

**GARCH Models**

- Modeling time-varying volatility
- Financial returns analysis
- Risk management applications
- Long memory processes

**Vector Autoregression (VAR)**

- Multiple time series modeling
- Granger causality testing
- Impulse response analysis
- Forecasting multiple variables

---

## Communication and Evaluation

### Effective Communication

**Audience Analysis**

- Understanding technical background
- Identifying key concerns and interests
- Adapting complexity to audience level
- Anticipating questions and objections

**Message Structure**

- Clear problem statement
- Logical flow of arguments
- Supporting evidence and examples
- Actionable conclusions

**Visual Communication**

- Appropriate chart and graph selection
- Clear labeling and legends
- Consistent formatting and style
- Storytelling with data

### Model Evaluation

**Performance Metrics**

- **Classification**: Accuracy, precision, recall, F1-score, AUC-ROC
- **Regression**: MSE, MAE, R-squared, adjusted R-squared
- **Time Series**: MAPE, RMSE, directional accuracy
- **Business Metrics**: ROI, customer lifetime value, conversion rates

**Validation Strategies**

- Train/validation/test splits
- Cross-validation techniques
- Out-of-time validation
- A/B testing in production

**Interpretability**

- Feature importance analysis
- Model explanations and visualizations
- Sensitivity analysis
- Business impact assessment

---

## Web Development

### Frontend Technologies

**HTML**

- Semantic markup for accessibility
- Form validation and user input
- SEO optimization
- Responsive design principles

**CSS**

- Layout systems (Flexbox, Grid)
- Responsive design with media queries
- CSS preprocessors (Sass, Less)
- Animation and transitions

**JavaScript**

- DOM manipulation and events
- Asynchronous programming (Promises, async/await)
- Modern ES6+ features
- Error handling and debugging

**React Framework**

- Component-based architecture
- State management (useState, useContext, Redux)
- Hooks and functional components
- Performance optimization

### Backend Considerations

**API Design**

- RESTful principles
- Authentication and authorization
- Error handling and status codes
- Documentation and versioning

**Database Design**

- Normalization and denormalization
- Indexing strategies
- Query optimization
- Data integrity constraints

---

## Emerging Trends and Future Directions

### Advanced AI Technologies

**Large Language Models (LLMs)**

- **Transformer Architecture**: Attention mechanisms enabling context understanding
- **Few-Shot Learning**: Learning from minimal examples
- **Prompt Engineering**: Designing effective inputs for desired outputs
- **Fine-tuning Strategies**: Adapting pre-trained models for specific domains
- **Multimodal Integration**: Combining text, image, and other data types

**Federated Learning**

- **Privacy-Preserving ML**: Training models without sharing raw data
- **Distributed Training**: Coordinating model updates across multiple devices
- **Communication Efficiency**: Minimizing data transfer between participants
- **Heterogeneous Data**: Handling different data distributions across participants
- **Security Considerations**: Protecting against adversarial attacks

**AutoML and Neural Architecture Search**

- **Automated Feature Engineering**: Discovering optimal feature combinations
- **Hyperparameter Optimization**: Efficient search for optimal model configurations
- **Neural Architecture Search**: Automatically designing neural network architectures
- **Multi-objective Optimization**: Balancing accuracy, speed, and resource usage
- **Interpretable AutoML**: Making automated decisions explainable

### Edge Computing and IoT

**Edge AI Deployment**

- **Model Compression**: Reducing model size for edge devices
- **Quantization**: Using lower precision for faster inference
- **Knowledge Distillation**: Transferring knowledge from large to small models
- **Hardware Acceleration**: Specialized chips for AI workloads
- **Energy Efficiency**: Optimizing for battery-powered devices

**IoT Data Analytics**

- **Stream Processing**: Real-time analysis of sensor data
- **Anomaly Detection**: Identifying unusual patterns in IoT data
- **Predictive Maintenance**: Forecasting equipment failures
- **Environmental Monitoring**: Tracking air quality, weather, and other conditions
- **Smart City Applications**: Traffic optimization, waste management, energy distribution

### Responsible AI and Ethics

**Bias and Fairness**

- **Algorithmic Bias Detection**: Identifying unfair treatment across groups
- **Fairness Metrics**: Quantifying different types of fairness
- **Debiasing Techniques**: Methods to reduce bias in models and data
- **Fairness-Aware Training**: Incorporating fairness constraints during training
- **Audit Frameworks**: Systematic evaluation of AI systems for bias

**Explainability and Transparency**

- **Model Interpretability**: Understanding how models make decisions
- **Explainable AI Techniques**: Methods for generating explanations
- **Regulatory Compliance**: Meeting requirements for AI transparency
- **User Trust**: Building confidence in AI systems
- **Stakeholder Communication**: Explaining AI decisions to different audiences

**Privacy and Security**

- **Differential Privacy**: Adding noise to protect individual privacy
- **Homomorphic Encryption**: Computing on encrypted data
- **Secure Multi-party Computation**: Collaborative analysis without sharing data
- **Adversarial Robustness**: Protecting against malicious inputs
- **AI Security**: Preventing attacks on AI systems themselves

### Quantum Computing and AI

**Quantum Machine Learning**

- **Quantum Algorithms**: Leveraging quantum properties for ML tasks
- **Quantum Feature Maps**: Encoding classical data in quantum states
- **Variational Quantum Circuits**: Hybrid classical-quantum approaches
- **Quantum Optimization**: Solving complex optimization problems
- **Quantum Neural Networks**: Neural networks implemented on quantum hardware

**Applications in Finance and Science**

- **Portfolio Optimization**: Finding optimal investment allocations
- **Risk Assessment**: Modeling complex financial scenarios
- **Drug Discovery**: Simulating molecular interactions
- **Climate Modeling**: Understanding complex environmental systems
- **Materials Science**: Discovering new materials with desired properties

### Sustainable AI

**Green AI Practices**

- **Energy-Efficient Training**: Reducing computational requirements
- **Model Efficiency**: Designing smaller, faster models
- **Renewable Energy**: Using sustainable energy sources for computation
- **Carbon Footprint Tracking**: Measuring environmental impact of AI systems
- **Sustainable Data Centers**: Optimizing infrastructure for energy efficiency

**AI for Sustainability**

- **Climate Change Modeling**: Predicting and mitigating climate impacts
- **Renewable Energy Optimization**: Maximizing efficiency of solar and wind power
- **Smart Grid Management**: Balancing energy supply and demand
- **Waste Reduction**: Optimizing resource usage and recycling
- **Biodiversity Conservation**: Monitoring and protecting ecosystems

### Future Research Directions

**Neuromorphic Computing**

- **Brain-Inspired Architectures**: Computing systems that mimic biological brains
- **Spiking Neural Networks**: Models that use temporal information
- **Event-Driven Processing**: Processing only when events occur
- **Low-Power Computing**: Energy-efficient neuromorphic chips
- **Adaptive Learning**: Systems that learn continuously from experience

**AI-Augmented Human Intelligence**

- **Human-AI Collaboration**: Systems that enhance human capabilities
- **Augmented Reality**: AI-powered visual and audio enhancements
- **Brain-Computer Interfaces**: Direct communication between brain and computers
- **Cognitive Enhancement**: Tools that improve human decision-making
- **Creative AI**: Systems that assist in artistic and creative endeavors

**Autonomous Systems**

- **Self-Driving Vehicles**: Full autonomy in transportation
- **Robotic Process Automation**: Automating complex business processes
- **Autonomous Drones**: Unmanned aerial vehicles for various applications
- **Smart Manufacturing**: Fully automated production systems
- **Autonomous Research**: AI systems that conduct scientific research

---

## Code Examples and Implementation Guides

### Python Implementation Examples

**Data Preprocessing and Feature Engineering**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Time-based feature engineering
def create_time_features(df, date_column):
    """Create comprehensive time-based features"""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Basic time features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter

    # Cyclical encoding for periodic features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    return df

# Lag feature creation for time series
def create_lag_features(df, target_column, lags=[1, 7, 30]):
    """Create lag features for time series prediction"""
    df = df.copy()
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

# Rolling statistics
def create_rolling_features(df, target_column, windows=[7, 14, 30]):
    """Create rolling window statistics"""
    df = df.copy()
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window).mean()
        df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window).std()
        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window).min()
        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window).max()
    return df
```

**Model Training and Validation Framework**

```python
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainingFramework:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def prepare_data(self, X, y, test_size=0.2, validation_size=0.2):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Second split: separate validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, model, X_train, y_train, model_name):
        """Train a model and store it"""
        model.fit(X_train, y_train)
        self.models[model_name] = model
        return model

    def evaluate_model(self, model, X, y, dataset_name):
        """Evaluate model performance"""
        y_pred = model.predict(X)

        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

        self.results[f"{model_name}_{dataset_name}"] = metrics
        return metrics

    def plot_results(self):
        """Plot model comparison results"""
        results_df = pd.DataFrame(self.results).T

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['mse', 'rmse', 'mae', 'r2']

        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            results_df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

# Usage example
def example_usage():
    # Sample data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.1

    # Initialize framework
    framework = ModelTrainingFramework()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = framework.prepare_data(X, y)

    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    framework.train_model(rf_model, X_train, y_train, 'RandomForest')

    # Evaluate
    framework.evaluate_model(rf_model, X_val, y_val, 'validation')
    framework.evaluate_model(rf_model, X_test, y_test, 'test')

    # Plot results
    framework.plot_results()
```

**Feature Selection and Dimensionality Reduction**

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

class FeatureSelectionFramework:
    def __init__(self):
        self.selected_features = {}
        self.feature_importance = {}

    def correlation_analysis(self, X, y, threshold=0.8):
        """Remove highly correlated features"""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above threshold
        high_corr_features = [column for column in upper_tri.columns
                             if any(upper_tri[column] > threshold)]

        X_filtered = X.drop(columns=high_corr_features)
        return X_filtered, high_corr_features

    def statistical_selection(self, X, y, k=10):
        """Select top k features using statistical tests"""
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]

        self.selected_features['statistical'] = selected_features
        self.feature_importance['statistical'] = dict(zip(selected_features, feature_scores))

        return X_selected, selected_features

    def recursive_feature_elimination(self, X, y, n_features=10):
        """Recursive feature elimination"""
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.support_].tolist()
        self.selected_features['rfe'] = selected_features

        return X_selected, selected_features

    def pca_analysis(self, X, n_components=0.95):
        """Principal Component Analysis"""
        if n_components < 1:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA(n_components=int(n_components))

        X_pca = pca.fit_transform(X)

        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        return X_pca, pca, explained_variance_ratio, cumulative_variance

    def plot_feature_importance(self, method='statistical'):
        """Plot feature importance"""
        if method not in self.feature_importance:
            print(f"No feature importance data for method: {method}")
            return

        importance_data = self.feature_importance[method]
        features = list(importance_data.keys())
        scores = list(importance_data.values())

        fig = go.Figure(data=[
            go.Bar(x=features, y=scores, marker_color='lightblue')
        ])

        fig.update_layout(
            title=f'Feature Importance - {method.title()} Method',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            xaxis_tickangle=-45
        )

        fig.show()
```

**Model Interpretability with SHAP**

```python
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class ModelInterpretability:
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = None

    def create_shap_explainer(self, explainer_type='tree'):
        """Create SHAP explainer for the model"""
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, self.X_train)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, self.X_train)

        return self.explainer

    def plot_feature_importance(self, max_display=20):
        """Plot SHAP feature importance"""
        if self.explainer is None:
            self.create_shap_explainer()

        shap_values = self.explainer.shap_values(self.X_test)
        shap.summary_plot(shap_values, self.X_test, max_display=max_display)

    def plot_individual_prediction(self, sample_idx=0):
        """Plot SHAP values for individual prediction"""
        if self.explainer is None:
            self.create_shap_explainer()

        shap_values = self.explainer.shap_values(self.X_test.iloc[sample_idx:sample_idx+1])
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            self.X_test.iloc[sample_idx],
            matplotlib=True
        )

    def plot_dependence_plots(self, feature_names, max_display=5):
        """Plot SHAP dependence plots for top features"""
        if self.explainer is None:
            self.create_shap_explainer()

        shap_values = self.explainer.shap_values(self.X_test)

        for i, feature in enumerate(feature_names[:max_display]):
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                feature, shap_values, self.X_test,
                interaction_index='auto'
            )
            plt.title(f'SHAP Dependence Plot: {feature}')
            plt.tight_layout()
            plt.show()

# Example usage
def interpretability_example():
    # Sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10),
                     columns=[f'feature_{i}' for i in range(10)])
    y = X['feature_0'] * 2 + X['feature_1'] * 3 + np.random.randn(1000) * 0.1

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create interpretability object
    interpreter = ModelInterpretability(model, X, X)

    # Plot feature importance
    interpreter.plot_feature_importance()

    # Plot individual prediction
    interpreter.plot_individual_prediction()

    # Plot dependence plots
    interpreter.plot_dependence_plots(['feature_0', 'feature_1', 'feature_2'])
```

**Time Series Analysis Implementation**

```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalysis:
    def __init__(self, data, date_column=None):
        self.data = data
        self.date_column = date_column
        if date_column:
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data = self.data.set_index(date_column)

    def check_stationarity(self, series, significance_level=0.05):
        """Check stationarity using ADF and KPSS tests"""
        # ADF Test
        adf_result = adfuller(series.dropna())
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]

        # KPSS Test
        kpss_result = kpss(series.dropna())
        kpss_statistic = kpss_result[0]
        kpss_pvalue = kpss_result[1]

        # Interpretation
        is_stationary_adf = adf_pvalue < significance_level
        is_stationary_kpss = kpss_pvalue > significance_level

        results = {
            'ADF_Statistic': adf_statistic,
            'ADF_pvalue': adf_pvalue,
            'ADF_Stationary': is_stationary_adf,
            'KPSS_Statistic': kpss_statistic,
            'KPSS_pvalue': kpss_pvalue,
            'KPSS_Stationary': is_stationary_kpss,
            'Overall_Stationary': is_stationary_adf and is_stationary_kpss
        }

        return results

    def make_stationary(self, series, max_diff=2):
        """Make series stationary through differencing"""
        diff_series = series.copy()
        diff_order = 0

        for i in range(max_diff):
            stationarity = self.check_stationarity(diff_series)
            if stationarity['Overall_Stationary']:
                break

            diff_series = diff_series.diff().dropna()
            diff_order += 1

        return diff_series, diff_order

    def plot_acf_pacf(self, series, lags=40):
        """Plot ACF and PACF for ARIMA model identification"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # ACF Plot
        plot_acf(series.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)')

        # PACF Plot
        plot_pacf(series.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        plt.show()

    def fit_arima(self, series, order, seasonal_order=None):
        """Fit ARIMA model"""
        if seasonal_order:
            model = sm.tsa.statespace.SARIMAX(
                series, order=order, seasonal_order=seasonal_order
            )
        else:
            model = ARIMA(series, order=order)

        fitted_model = model.fit()
        return fitted_model

    def forecast_arima(self, model, steps=12):
        """Generate forecasts using fitted ARIMA model"""
        forecast = model.forecast(steps=steps)
        forecast_ci = model.get_forecast(steps=steps).conf_int()

        return forecast, forecast_ci

    def evaluate_forecast(self, actual, predicted):
        """Evaluate forecast accuracy"""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

        return metrics

# Example usage
def time_series_example():
    # Generate sample time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    trend = np.linspace(0, 10, 1000)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
    noise = np.random.normal(0, 0.5, 1000)

    series = trend + seasonal + noise
    ts_data = pd.DataFrame({'value': series}, index=dates)

    # Initialize time series analysis
    ts_analysis = TimeSeriesAnalysis(ts_data)

    # Check stationarity
    stationarity = ts_analysis.check_stationarity(ts_data['value'])
    print("Stationarity Test Results:", stationarity)

    # Make stationary if needed
    if not stationarity['Overall_Stationary']:
        stationary_series, diff_order = ts_analysis.make_stationary(ts_data['value'])
        print(f"Series made stationary with {diff_order} differences")
    else:
        stationary_series = ts_data['value']
        diff_order = 0

    # Plot ACF and PACF
    ts_analysis.plot_acf_pacf(stationary_series)

    # Fit ARIMA model (example order)
    model = ts_analysis.fit_arima(stationary_series, order=(1, diff_order, 1))
    print("ARIMA Model Summary:")
    print(model.summary())

    # Generate forecasts
    forecast, ci = ts_analysis.forecast_arima(model, steps=30)
    print("Forecast generated for next 30 periods")
```

**Statistical Analysis Implementation**

```python
from scipy import stats
from scipy.stats import norm, t, chi2, f
import pingouin as pg

class StatisticalAnalysis:
    def __init__(self):
        self.results = {}

    def descriptive_statistics(self, data):
        """Calculate comprehensive descriptive statistics"""
        stats_dict = {
            'count': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),
            'variance': np.var(data, ddof=1),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }

        return stats_dict

    def normality_tests(self, data):
        """Perform multiple normality tests"""
        tests = {
            'Shapiro-Wilk': stats.shapiro(data),
            'Anderson-Darling': stats.anderson(data),
            'Kolmogorov-Smirnov': stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        }

        return tests

    def correlation_analysis(self, x, y, method='pearson'):
        """Perform correlation analysis with confidence intervals"""
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)

        # Calculate confidence interval using Fisher's z-transformation
        z = np.arctanh(corr)
        se = 1 / np.sqrt(len(x) - 3)
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)

        results = {
            'correlation': corr,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': method
        }

        return results

    def hypothesis_testing(self, data1, data2=None, test_type='one_sample', alpha=0.05):
        """Perform various hypothesis tests"""
        if test_type == 'one_sample':
            # One-sample t-test
            statistic, p_value = stats.ttest_1samp(data1, 0)
            test_name = 'One-Sample t-test'

        elif test_type == 'two_sample':
            # Two-sample t-test
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_name = 'Two-Sample t-test'

        elif test_type == 'paired':
            # Paired t-test
            statistic, p_value = stats.ttest_rel(data1, data2)
            test_name = 'Paired t-test'

        elif test_type == 'mann_whitney':
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = 'Mann-Whitney U test'

        # Calculate effect size (Cohen's d for t-tests)
        if 't-test' in test_name:
            if test_type == 'one_sample':
                effect_size = np.mean(data1) / np.std(data1, ddof=1)
            else:
                pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                                    (len(data2) - 1) * np.var(data2, ddof=1)) /
                                   (len(data1) + len(data2) - 2))
                effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        else:
            effect_size = None

        results = {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': effect_size,
            'alpha': alpha
        }

        return results

    def power_analysis(self, effect_size, alpha=0.05, power=0.8, ratio=1.0):
        """Perform power analysis for sample size determination"""
        from statsmodels.stats.power import TTestPower

        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio
        )

        return int(sample_size)

    def bootstrap_confidence_interval(self, data, statistic_func, n_bootstrap=10000, confidence_level=0.95):
        """Calculate bootstrap confidence interval"""
        bootstrap_statistics = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic_func(bootstrap_sample))

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)

        return ci_lower, ci_upper

# Example usage
def statistical_analysis_example():
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(100, 15, 50)
    group2 = np.random.normal(105, 15, 50)

    # Initialize statistical analysis
    stats_analysis = StatisticalAnalysis()

    # Descriptive statistics
    desc_stats = stats_analysis.descriptive_statistics(group1)
    print("Descriptive Statistics for Group 1:")
    for key, value in desc_stats.items():
        print(f"{key}: {value:.4f}")

    # Normality tests
    normality = stats_analysis.normality_tests(group1)
    print("\nNormality Tests:")
    for test_name, (statistic, p_value) in normality.items():
        print(f"{test_name}: statistic={statistic:.4f}, p-value={p_value:.4f}")

    # Correlation analysis
    correlation = stats_analysis.correlation_analysis(group1, group2)
    print(f"\nCorrelation Analysis:")
    print(f"Correlation: {correlation['correlation']:.4f}")
    print(f"P-value: {correlation['p_value']:.4f}")
    print(f"95% CI: [{correlation['ci_lower']:.4f}, {correlation['ci_upper']:.4f}]")

    # Hypothesis testing
    t_test = stats_analysis.hypothesis_testing(group1, group2, test_type='two_sample')
    print(f"\nHypothesis Test Results:")
    print(f"Test: {t_test['test_name']}")
    print(f"Statistic: {t_test['statistic']:.4f}")
    print(f"P-value: {t_test['p_value']:.4f}")
    print(f"Significant: {t_test['significant']}")
    print(f"Effect Size: {t_test['effect_size']:.4f}")
```

### R Implementation Examples

**Statistical Analysis in R**

```r
# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# Data preprocessing function
preprocess_data <- function(data, target_col, test_size = 0.2) {
  # Remove missing values
  data_clean <- data %>% drop_na()

  # Split data
  set.seed(42)
  train_index <- createDataPartition(data_clean[[target_col]],
                                    p = 1 - test_size,
                                    list = FALSE)
  train_data <- data_clean[train_index, ]
  test_data <- data_clean[-train_index, ]

  return(list(train = train_data, test = test_data))
}

# Feature engineering function
create_features <- function(data) {
  data %>%
    mutate(
      # Time-based features
      year = year(date_column),
      month = month(date_column),
      day_of_week = wday(date_column),

      # Cyclical encoding
      month_sin = sin(2 * pi * month / 12),
      month_cos = cos(2 * pi * month / 12),

      # Lag features
      lag_1 = lag(target_column, 1),
      lag_7 = lag(target_column, 7),

      # Rolling statistics
      rolling_mean_7 = rollmean(target_column, 7, fill = NA, align = "right"),
      rolling_std_7 = rollapply(target_column, 7, sd, fill = NA, align = "right")
    )
}

# Model training and evaluation
train_evaluate_model <- function(train_data, test_data, target_col) {
  # Prepare formula
  formula <- as.formula(paste(target_col, "~ ."))

  # Train random forest
  rf_model <- randomForest(formula, data = train_data, ntree = 100)

  # Make predictions
  train_pred <- predict(rf_model, train_data)
  test_pred <- predict(rf_model, test_data)

  # Calculate metrics
  train_rmse <- sqrt(mean((train_data[[target_col]] - train_pred)^2))
  test_rmse <- sqrt(mean((test_data[[target_col]] - test_pred)^2))

  # Feature importance
  importance_df <- data.frame(
    feature = rownames(importance(rf_model)),
    importance = importance(rf_model)[, 1]
  ) %>%
    arrange(desc(importance))

  return(list(
    model = rf_model,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    importance = importance_df
  ))
}

# Visualization functions
plot_feature_importance <- function(importance_df, top_n = 10) {
  importance_df %>%
    head(top_n) %>%
    ggplot(aes(x = reorder(feature, importance), y = importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Feature Importance",
         x = "Features",
         y = "Importance") +
    theme_minimal()
}

plot_predictions <- function(actual, predicted, title = "Actual vs Predicted") {
  data.frame(actual = actual, predicted = predicted) %>%
    ggplot(aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = title,
         x = "Actual Values",
         y = "Predicted Values") +
    theme_minimal()
}
```

### SQL Implementation Examples

**Data Analysis Queries**

```sql
-- Time series analysis with window functions
WITH daily_metrics AS (
  SELECT
    DATE(created_at) as date,
    COUNT(*) as daily_orders,
    SUM(amount) as daily_revenue,
    AVG(amount) as avg_order_value
  FROM orders
  WHERE created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
  GROUP BY DATE(created_at)
),
rolling_stats AS (
  SELECT
    date,
    daily_orders,
    daily_revenue,
    avg_order_value,
    -- 7-day rolling averages
    AVG(daily_orders) OVER (
      ORDER BY date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_avg_orders_7d,
    AVG(daily_revenue) OVER (
      ORDER BY date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_avg_revenue_7d,
    -- 30-day rolling averages
    AVG(daily_orders) OVER (
      ORDER BY date
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_avg_orders_30d,
    -- Growth rates
    LAG(daily_revenue, 1) OVER (ORDER BY date) as prev_day_revenue,
    LAG(daily_revenue, 7) OVER (ORDER BY date) as prev_week_revenue
  FROM daily_metrics
)
SELECT
  date,
  daily_orders,
  daily_revenue,
  avg_order_value,
  rolling_avg_orders_7d,
  rolling_avg_revenue_7d,
  rolling_avg_orders_30d,
  -- Daily growth rate
  CASE
    WHEN prev_day_revenue > 0
    THEN (daily_revenue - prev_day_revenue) / prev_day_revenue * 100
    ELSE NULL
  END as daily_growth_rate,
  -- Weekly growth rate
  CASE
    WHEN prev_week_revenue > 0
    THEN (daily_revenue - prev_week_revenue) / prev_week_revenue * 100
    ELSE NULL
  END as weekly_growth_rate
FROM rolling_stats
ORDER BY date DESC;

-- Customer segmentation analysis
WITH customer_metrics AS (
  SELECT
    customer_id,
    COUNT(*) as total_orders,
    SUM(amount) as total_spent,
    AVG(amount) as avg_order_value,
    MIN(created_at) as first_order,
    MAX(created_at) as last_order,
    DATEDIFF(MAX(created_at), MIN(created_at)) as customer_lifetime_days
  FROM orders
  WHERE created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 365 DAY)
  GROUP BY customer_id
),
customer_segments AS (
  SELECT
    customer_id,
    total_orders,
    total_spent,
    avg_order_value,
    customer_lifetime_days,
    -- RFM Segmentation
    CASE
      WHEN total_spent >= 1000 THEN 'High Value'
      WHEN total_spent >= 500 THEN 'Medium Value'
      ELSE 'Low Value'
    END as value_segment,
    CASE
      WHEN total_orders >= 10 THEN 'Frequent'
      WHEN total_orders >= 5 THEN 'Regular'
      ELSE 'Occasional'
    END as frequency_segment,
    CASE
      WHEN DATEDIFF(CURRENT_DATE, last_order) <= 30 THEN 'Recent'
      WHEN DATEDIFF(CURRENT_DATE, last_order) <= 90 THEN 'Active'
      ELSE 'Inactive'
    END as recency_segment
  FROM customer_metrics
)
SELECT
  value_segment,
  frequency_segment,
  recency_segment,
  COUNT(*) as customer_count,
  AVG(total_spent) as avg_total_spent,
  AVG(total_orders) as avg_total_orders,
  AVG(avg_order_value) as avg_order_value
FROM customer_segments
GROUP BY value_segment, frequency_segment, recency_segment
ORDER BY avg_total_spent DESC;

-- A/B Testing Analysis
WITH ab_test_results AS (
  SELECT
    user_id,
    variant,
    conversion_flag,
    revenue,
    session_duration,
    page_views
  FROM ab_test_data
  WHERE test_name = 'homepage_redesign'
    AND created_at >= '2024-01-01'
    AND created_at < '2024-02-01'
),
variant_stats AS (
  SELECT
    variant,
    COUNT(*) as total_users,
    SUM(conversion_flag) as conversions,
    AVG(conversion_flag) as conversion_rate,
    AVG(revenue) as avg_revenue,
    AVG(session_duration) as avg_session_duration,
    AVG(page_views) as avg_page_views,
    SUM(revenue) as total_revenue
  FROM ab_test_results
  GROUP BY variant
),
statistical_test AS (
  SELECT
    a.variant as variant_a,
    b.variant as variant_b,
    a.conversion_rate as rate_a,
    b.conversion_rate as rate_b,
    a.total_users as n_a,
    b.total_users as n_b,
    -- Z-test for proportion difference
    (a.conversion_rate - b.conversion_rate) /
    SQRT(
      (a.conversion_rate * (1 - a.conversion_rate) / a.total_users) +
      (b.conversion_rate * (1 - b.conversion_rate) / b.total_users)
    ) as z_statistic
  FROM variant_stats a
  CROSS JOIN variant_stats b
  WHERE a.variant < b.variant
)
SELECT
  variant_a,
  variant_b,
  rate_a,
  rate_b,
  rate_a - rate_b as rate_difference,
  z_statistic,
  -- P-value approximation (two-tailed test)
  2 * (1 - ABS(NORMSDIST(z_statistic))) as p_value,
  CASE
    WHEN ABS(z_statistic) > 1.96 THEN 'Significant'
    ELSE 'Not Significant'
  END as significance
FROM statistical_test;
```

---

## Finance Domain Examples and Implementations

### Quantitative Finance and Risk Management

**Portfolio Optimization and Asset Allocation**

```python
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.cov_matrix = None

    def fetch_data(self):
        """Fetch historical price data for portfolio optimization"""
        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                data[ticker] = hist['Close']
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

        self.data = pd.DataFrame(data)
        self.returns = self.data.pct_change().dropna()
        self.cov_matrix = self.returns.cov()
        return self.data, self.returns

    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252  # Annualized
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol != 0 else 0

        return portfolio_return, portfolio_vol, sharpe_ratio

    def optimize_portfolio(self, objective='sharpe', risk_free_rate=0.02):
        """Optimize portfolio weights using different objectives"""
        n_assets = len(self.tickers)

        # Constraints: weights sum to 1, no short selling
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial weights (equal weight)
        initial_weights = np.array([1/n_assets] * n_assets)

        if objective == 'sharpe':
            # Maximize Sharpe ratio
            def objective_function(weights):
                return -self.calculate_portfolio_metrics(weights)[2]
        elif objective == 'min_vol':
            # Minimize volatility
            def objective_function(weights):
                return self.calculate_portfolio_metrics(weights)[1]
        elif objective == 'max_return':
            # Maximize return
            def objective_function(weights):
                return -self.calculate_portfolio_metrics(weights)[0]

        # Optimize
        result = minimize(objective_function, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x, result.success

    def efficient_frontier(self, num_portfolios=1000):
        """Generate efficient frontier"""
        returns_list = []
        volatility_list = []
        sharpe_list = []
        weights_list = []

        for _ in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights = weights / np.sum(weights)

            portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(weights)

            returns_list.append(portfolio_return)
            volatility_list.append(portfolio_vol)
            sharpe_list.append(sharpe)
            weights_list.append(weights)

        return pd.DataFrame({
            'Return': returns_list,
            'Volatility': volatility_list,
            'Sharpe': sharpe_list,
            'Weights': weights_list
        })

    def plot_efficient_frontier(self, efficient_frontier_df):
        """Plot efficient frontier with optimal portfolios"""
        plt.figure(figsize=(12, 8))

        # Plot random portfolios
        plt.scatter(efficient_frontier_df['Volatility'], efficient_frontier_df['Return'],
                   c=efficient_frontier_df['Sharpe'], cmap='viridis', alpha=0.6)

        # Find optimal portfolios
        max_sharpe_idx = efficient_frontier_df['Sharpe'].idxmax()
        min_vol_idx = efficient_frontier_df['Volatility'].idxmin()

        # Plot optimal points
        plt.scatter(efficient_frontier_df.loc[max_sharpe_idx, 'Volatility'],
                   efficient_frontier_df.loc[max_sharpe_idx, 'Return'],
                   color='red', s=200, marker='*', label='Maximum Sharpe Ratio')

        plt.scatter(efficient_frontier_df.loc[min_vol_idx, 'Volatility'],
                   efficient_frontier_df.loc[min_vol_idx, 'Return'],
                   color='green', s=200, marker='*', label='Minimum Volatility')

        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Portfolio Volatility')
        plt.ylabel('Portfolio Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.show()

# Example usage
def portfolio_optimization_example():
    # Define portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG']
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    # Initialize optimizer
    optimizer = PortfolioOptimizer(tickers, start_date, end_date)
    data, returns = optimizer.fetch_data()

    # Generate efficient frontier
    efficient_frontier = optimizer.efficient_frontier()
    optimizer.plot_efficient_frontier(efficient_frontier)

    # Optimize for maximum Sharpe ratio
    optimal_weights, success = optimizer.optimize_portfolio(objective='sharpe')

    if success:
        print("Optimal Portfolio Weights:")
        for ticker, weight in zip(tickers, optimal_weights):
            print(f"{ticker}: {weight:.4f}")

        # Calculate optimal portfolio metrics
        opt_return, opt_vol, opt_sharpe = optimizer.calculate_portfolio_metrics(optimal_weights)
        print(f"\nOptimal Portfolio Metrics:")
        print(f"Expected Return: {opt_return:.4f}")
        print(f"Volatility: {opt_vol:.4f}")
        print(f"Sharpe Ratio: {opt_sharpe:.4f}")
```

**Risk Management and Value at Risk (VaR)**

```python
import scipy.stats as stats
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    def __init__(self, returns, confidence_level=0.95):
        self.returns = returns
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def historical_var(self, portfolio_value=1000000):
        """Calculate Historical VaR"""
        var_percentile = np.percentile(self.returns, self.alpha * 100)
        var_dollar = portfolio_value * abs(var_percentile)

        return {
            'VaR_Percentage': var_percentile,
            'VaR_Dollar': var_dollar,
            'Confidence_Level': self.confidence_level
        }

    def parametric_var(self, portfolio_value=1000000, distribution='normal'):
        """Calculate Parametric VaR"""
        if distribution == 'normal':
            var_percentile = norm.ppf(self.alpha,
                                     loc=self.returns.mean(),
                                     scale=self.returns.std())
        elif distribution == 't':
            # Fit t-distribution
            df, loc, scale = t.fit(self.returns)
            var_percentile = t.ppf(self.alpha, df, loc=loc, scale=scale)

        var_dollar = portfolio_value * abs(var_percentile)

        return {
            'VaR_Percentage': var_percentile,
            'VaR_Dollar': var_dollar,
            'Distribution': distribution,
            'Confidence_Level': self.confidence_level
        }

    def monte_carlo_var(self, portfolio_value=1000000, n_simulations=10000):
        """Calculate Monte Carlo VaR"""
        # Fit normal distribution to returns
        mu, sigma = self.returns.mean(), self.returns.std()

        # Generate random returns
        simulated_returns = np.random.normal(mu, sigma, n_simulations)

        # Calculate VaR
        var_percentile = np.percentile(simulated_returns, self.alpha * 100)
        var_dollar = portfolio_value * abs(var_percentile)

        return {
            'VaR_Percentage': var_percentile,
            'VaR_Dollar': var_dollar,
            'Simulations': n_simulations,
            'Confidence_Level': self.confidence_level
        }

    def conditional_var(self, portfolio_value=1000000):
        """Calculate Conditional VaR (Expected Shortfall)"""
        var_threshold = np.percentile(self.returns, self.alpha * 100)
        tail_returns = self.returns[self.returns <= var_threshold]

        cvar_percentile = tail_returns.mean()
        cvar_dollar = portfolio_value * abs(cvar_percentile)

        return {
            'CVaR_Percentage': cvar_percentile,
            'CVaR_Dollar': cvar_dollar,
            'Confidence_Level': self.confidence_level
        }

    def stress_testing(self, portfolio_value=1000000, scenarios=None):
        """Perform stress testing with different scenarios"""
        if scenarios is None:
            scenarios = {
                'Market_Crash': -0.20,  # 20% market decline
                'Interest_Rate_Shock': -0.10,  # 10% decline
                'Volatility_Spike': -0.15,  # 15% decline
                'Sector_Decline': -0.12   # 12% decline
            }

        stress_results = {}
        for scenario, shock in scenarios.items():
            stressed_return = self.returns.mean() + shock
            stress_results[scenario] = {
                'Stressed_Return': stressed_return,
                'Portfolio_Impact': portfolio_value * stressed_return
            }

        return stress_results

    def backtest_var(self, window=252):
        """Backtest VaR predictions"""
        var_predictions = []
        actual_returns = []

        for i in range(window, len(self.returns)):
            # Calculate VaR using historical data up to point i
            historical_data = self.returns.iloc[:i]
            risk_manager = RiskManager(historical_data, self.confidence_level)
            var_result = risk_manager.historical_var()

            var_predictions.append(var_result['VaR_Percentage'])
            actual_returns.append(self.returns.iloc[i])

        # Calculate violation rate
        violations = sum(1 for actual, var in zip(actual_returns, var_predictions)
                        if actual < var)
        violation_rate = violations / len(actual_returns)

        return {
            'Violation_Rate': violation_rate,
            'Expected_Violation_Rate': self.alpha,
            'VaR_Predictions': var_predictions,
            'Actual_Returns': actual_returns
        }

# Example usage
def risk_management_example():
    # Generate sample portfolio returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.02, 1000), index=dates)

    # Initialize risk manager
    risk_manager = RiskManager(returns, confidence_level=0.95)

    # Calculate different VaR measures
    historical_var = risk_manager.historical_var()
    parametric_var = risk_manager.parametric_var()
    monte_carlo_var = risk_manager.monte_carlo_var()
    conditional_var = risk_manager.conditional_var()

    print("Risk Management Results:")
    print(f"Historical VaR: ${historical_var['VaR_Dollar']:,.2f}")
    print(f"Parametric VaR: ${parametric_var['VaR_Dollar']:,.2f}")
    print(f"Monte Carlo VaR: ${monte_carlo_var['VaR_Dollar']:,.2f}")
    print(f"Conditional VaR: ${conditional_var['CVaR_Dollar']:,.2f}")

    # Stress testing
    stress_results = risk_manager.stress_testing()
    print("\nStress Testing Results:")
    for scenario, result in stress_results.items():
        print(f"{scenario}: ${result['Portfolio_Impact']:,.2f}")

    # Backtest VaR
    backtest_results = risk_manager.backtest_var()
    print(f"\nVaR Backtest:")
    print(f"Violation Rate: {backtest_results['Violation_Rate']:.4f}")
    print(f"Expected Rate: {backtest_results['Expected_Violation_Rate']:.4f}")
```

**Options Pricing and Greeks**

```python
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class OptionsPricer:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

    def black_scholes(self):
        """Calculate Black-Scholes option price and Greeks"""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) -
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:  # put
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) +
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))

        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1)
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) if self.option_type == 'call' else -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        return {
            'Price': price,
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }

    def binomial_tree(self, n_steps=100):
        """Calculate option price using binomial tree"""
        dt = self.T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)

        # Create stock price tree
        stock_prices = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_prices[i, j] = self.S * (u ** (i - j)) * (d ** j)

        # Calculate option prices at expiration
        option_prices = np.zeros((n_steps + 1, n_steps + 1))
        for j in range(n_steps + 1):
            if self.option_type == 'call':
                option_prices[n_steps, j] = max(0, stock_prices[n_steps, j] - self.K)
            else:
                option_prices[n_steps, j] = max(0, self.K - stock_prices[n_steps, j])

        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_prices[i, j] = np.exp(-self.r * dt) * (p * option_prices[i + 1, j] +
                                                             (1 - p) * option_prices[i + 1, j + 1])

        return option_prices[0, 0]

    def monte_carlo_pricing(self, n_simulations=10000):
        """Calculate option price using Monte Carlo simulation"""
        # Generate random stock price paths
        Z = np.random.normal(0, 1, n_simulations)
        S_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T +
                              self.sigma * np.sqrt(self.T) * Z)

        # Calculate option payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)

        # Discount payoffs
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        option_std = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_simulations)

        return {
            'Price': option_price,
            'Standard_Error': option_std,
            'Confidence_Interval': (option_price - 1.96 * option_std,
                                  option_price + 1.96 * option_std)
        }

    def plot_greeks(self, param_range, param_name):
        """Plot option Greeks as a function of a parameter"""
        prices = []
        deltas = []
        gammas = []
        thetas = []
        vegas = []

        for param in param_range:
            if param_name == 'S':
                pricer = OptionsPricer(param, self.K, self.T, self.r, self.sigma, self.option_type)
            elif param_name == 'K':
                pricer = OptionsPricer(self.S, param, self.T, self.r, self.sigma, self.option_type)
            elif param_name == 'T':
                pricer = OptionsPricer(self.S, self.K, param, self.r, self.sigma, self.option_type)
            elif param_name == 'sigma':
                pricer = OptionsPricer(self.S, self.K, self.T, self.r, param, self.option_type)

            result = pricer.black_scholes()
            prices.append(result['Price'])
            deltas.append(result['Delta'])
            gammas.append(result['Gamma'])
            thetas.append(result['Theta'])
            vegas.append(result['Vega'])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Option Greeks vs {param_name}')

        axes[0, 0].plot(param_range, prices)
        axes[0, 0].set_title('Price')
        axes[0, 0].set_xlabel(param_name)

        axes[0, 1].plot(param_range, deltas)
        axes[0, 1].set_title('Delta')
        axes[0, 1].set_xlabel(param_name)

        axes[0, 2].plot(param_range, gammas)
        axes[0, 2].set_title('Gamma')
        axes[0, 2].set_xlabel(param_name)

        axes[1, 0].plot(param_range, thetas)
        axes[1, 0].set_title('Theta')
        axes[1, 0].set_xlabel(param_name)

        axes[1, 1].plot(param_range, vegas)
        axes[1, 1].set_title('Vega')
        axes[1, 1].set_xlabel(param_name)

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

# Example usage
def options_pricing_example():
    # Define option parameters
    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1.0  # Time to expiration (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    # Create option pricer
    call_option = OptionsPricer(S, K, T, r, sigma, 'call')
    put_option = OptionsPricer(S, K, T, r, sigma, 'put')

    # Calculate Black-Scholes prices and Greeks
    call_result = call_option.black_scholes()
    put_result = put_option.black_scholes()

    print("Call Option Results:")
    for key, value in call_result.items():
        print(f"{key}: {value:.4f}")

    print("\nPut Option Results:")
    for key, value in put_result.items():
        print(f"{key}: {value:.4f}")

    # Compare different pricing methods
    bs_price = call_result['Price']
    binomial_price = call_option.binomial_tree()
    mc_result = call_option.monte_carlo_pricing()

    print(f"\nPricing Method Comparison:")
    print(f"Black-Scholes: {bs_price:.4f}")
    print(f"Binomial Tree: {binomial_price:.4f}")
    print(f"Monte Carlo: {mc_result['Price']:.4f} ± {mc_result['Standard_Error']:.4f}")

    # Plot Greeks
    stock_prices = np.linspace(80, 120, 100)
    call_option.plot_greeks(stock_prices, 'S')
```

**Credit Risk Modeling**

```python
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def generate_sample_data(self, n_samples=10000):
        """Generate sample credit data for modeling"""
        np.random.seed(42)

        # Generate features
        data = {
            'age': np.random.normal(45, 15, n_samples),
            'income': np.random.lognormal(10, 0.5, n_samples),
            'credit_score': np.random.normal(700, 100, n_samples),
            'debt_to_income': np.random.beta(2, 5, n_samples) * 2,
            'payment_history': np.random.beta(8, 2, n_samples),
            'employment_length': np.random.exponential(5, n_samples),
            'loan_amount': np.random.lognormal(10, 0.3, n_samples),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples)
        }

        df = pd.DataFrame(data)

        # Create target variable (default probability)
        # Higher risk for: low income, high debt, poor payment history, low credit score
        risk_score = (
            -0.1 * (df['income'] - df['income'].mean()) / df['income'].std() +
            0.3 * (df['debt_to_income'] - df['debt_to_income'].mean()) / df['debt_to_income'].std() +
            -0.2 * (df['payment_history'] - df['payment_history'].mean()) / df['payment_history'].std() +
            -0.15 * (df['credit_score'] - df['credit_score'].mean()) / df['credit_score'].std() +
            np.random.normal(0, 0.1, n_samples)
        )

        # Convert to default probability
        default_prob = 1 / (1 + np.exp(-risk_score))
        df['default'] = np.random.binomial(1, default_prob)

        return df

    def train_model(self, X, y, model_type='logistic'):
        """Train credit risk model"""
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()

        return self.model

    def predict_default_probability(self, X):
        """Predict default probability"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict_proba(X)[:, 1]

    def calculate_credit_score(self, default_prob):
        """Convert default probability to credit score (300-850 scale)"""
        # Use logit transformation to convert probability to score
        score = 850 - 550 * np.log(default_prob / (1 - default_prob))
        return np.clip(score, 300, 850)

    def portfolio_credit_risk(self, loan_data, confidence_level=0.95):
        """Calculate portfolio-level credit risk metrics"""
        # Predict default probabilities
        X = loan_data.drop('default', axis=1, errors='ignore')
        default_probs = self.predict_default_probability(X)

        # Calculate expected loss
        loan_amounts = loan_data['loan_amount']
        expected_loss = np.sum(default_probs * loan_amounts)

        # Calculate Value at Risk (VaR) for credit portfolio
        # Simulate portfolio losses
        n_simulations = 10000
        portfolio_losses = []

        for _ in range(n_simulations):
            # Simulate defaults based on probabilities
            defaults = np.random.binomial(1, default_probs)
            portfolio_loss = np.sum(defaults * loan_amounts)
            portfolio_losses.append(portfolio_loss)

        var_threshold = np.percentile(portfolio_losses, (1 - confidence_level) * 100)

        # Calculate Expected Shortfall (Conditional VaR)
        tail_losses = [loss for loss in portfolio_losses if loss >= var_threshold]
        expected_shortfall = np.mean(tail_losses)

        return {
            'Expected_Loss': expected_loss,
            'VaR': var_threshold,
            'Expected_Shortfall': expected_shortfall,
            'Portfolio_Value': np.sum(loan_amounts),
            'Average_Default_Rate': np.mean(default_probs)
        }

    def stress_testing(self, loan_data, scenarios):
        """Perform stress testing on credit portfolio"""
        base_risk = self.portfolio_credit_risk(loan_data)
        stress_results = {}

        for scenario_name, scenario_params in scenarios.items():
            # Apply stress factors to loan data
            stressed_data = loan_data.copy()

            for param, factor in scenario_params.items():
                if param in stressed_data.columns:
                    stressed_data[param] = stressed_data[param] * factor

            # Recalculate risk metrics
            stressed_risk = self.portfolio_credit_risk(stressed_data)
            stress_results[scenario_name] = {
                'Expected_Loss': stressed_risk['Expected_Loss'],
                'VaR': stressed_risk['VaR'],
                'Loss_Increase': stressed_risk['Expected_Loss'] / base_risk['Expected_Loss'] - 1
            }

        return stress_results

    def plot_credit_distribution(self, loan_data):
        """Plot credit score distribution and risk metrics"""
        X = loan_data.drop('default', axis=1, errors='ignore')
        default_probs = self.predict_default_probability(X)
        credit_scores = self.calculate_credit_score(default_probs)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Credit score distribution
        axes[0, 0].hist(credit_scores, bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Credit Score Distribution')
        axes[0, 0].set_xlabel('Credit Score')
        axes[0, 0].set_ylabel('Frequency')

        # Default probability distribution
        axes[0, 1].hist(default_probs, bins=50, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Default Probability Distribution')
        axes[0, 1].set_xlabel('Default Probability')
        axes[0, 1].set_ylabel('Frequency')

        # Credit score vs default probability
        axes[1, 0].scatter(credit_scores, default_probs, alpha=0.5)
        axes[1, 0].set_title('Credit Score vs Default Probability')
        axes[1, 0].set_xlabel('Credit Score')
        axes[1, 0].set_ylabel('Default Probability')

        # Risk segments
        risk_segments = pd.cut(credit_scores, bins=[300, 580, 670, 740, 850],
                              labels=['Poor', 'Fair', 'Good', 'Excellent'])
        segment_counts = risk_segments.value_counts()

        axes[1, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Risk Segment Distribution')

        plt.tight_layout()
        plt.show()

# Example usage
def credit_risk_example():
    # Generate sample data
    credit_model = CreditRiskModel()
    loan_data = credit_model.generate_sample_data(5000)

    # Prepare features
    feature_cols = ['age', 'income', 'credit_score', 'debt_to_income',
                   'payment_history', 'employment_length', 'loan_amount', 'loan_term']
    X = loan_data[feature_cols]
    y = loan_data['default']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    credit_model.train_model(X_train, y_train, model_type='random_forest')

    # Evaluate model
    y_pred_proba = credit_model.predict_default_probability(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("Credit Risk Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Portfolio risk analysis
    portfolio_risk = credit_model.portfolio_credit_risk(loan_data)
    print(f"\nPortfolio Risk Metrics:")
    for key, value in portfolio_risk.items():
        print(f"{key}: ${value:,.2f}" if 'Loss' in key or 'VaR' in key or 'Value' in key else f"{key}: {value:.4f}")

    # Stress testing
    scenarios = {
        'Economic_Recession': {'income': 0.8, 'credit_score': 0.9},
        'Interest_Rate_Increase': {'debt_to_income': 1.2},
        'Unemployment_Spike': {'income': 0.7, 'payment_history': 0.8}
    }

    stress_results = credit_model.stress_testing(loan_data, scenarios)
    print(f"\nStress Testing Results:")
    for scenario, results in stress_results.items():
        print(f"{scenario}:")
        print(f"  Expected Loss: ${results['Expected_Loss']:,.2f}")
        print(f"  VaR: ${results['VaR']:,.2f}")
        print(f"  Loss Increase: {results['Loss_Increase']:.2%}")

    # Plot distributions
    credit_model.plot_credit_distribution(loan_data)
```

**Algorithmic Trading Strategy**

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    def __init__(self, ticker, start_date, end_date, initial_capital=100000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.positions = None
        self.portfolio_value = None

    def fetch_data(self):
        """Fetch historical price data"""
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        self.data['Returns'] = self.data['Close'].pct_change()
        return self.data

    def moving_average_crossover(self, short_window=20, long_window=50):
        """Moving average crossover strategy"""
        self.data['SMA_Short'] = self.data['Close'].rolling(window=short_window).mean()
        self.data['SMA_Long'] = self.data['Close'].rolling(window=long_window).mean()

        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['SMA_Short'] > self.data['SMA_Long'], 'Signal'] = 1
        self.data.loc[self.data['SMA_Short'] < self.data['SMA_Long'], 'Signal'] = -1

        # Calculate positions
        self.data['Position'] = self.data['Signal'].shift(1)

        return self.data

    def mean_reversion_strategy(self, window=20, std_dev=2):
        """Mean reversion strategy based on Bollinger Bands"""
        self.data['SMA'] = self.data['Close'].rolling(window=window).mean()
        self.data['STD'] = self.data['Close'].rolling(window=window).std()
        self.data['Upper_Band'] = self.data['SMA'] + (std_dev * self.data['STD'])
        self.data['Lower_Band'] = self.data['SMA'] - (std_dev * self.data['STD'])

        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['Close'] < self.data['Lower_Band'], 'Signal'] = 1  # Buy
        self.data.loc[self.data['Close'] > self.data['Upper_Band'], 'Signal'] = -1  # Sell

        # Calculate positions
        self.data['Position'] = self.data['Signal'].shift(1)

        return self.data

    def momentum_strategy(self, lookback_period=20, threshold=0.02):
        """Momentum-based trading strategy"""
        self.data['Returns_Lookback'] = self.data['Close'].pct_change(lookback_period)

        # Generate signals
        self.data['Signal'] = 0
        self.data.loc[self.data['Returns_Lookback'] > threshold, 'Signal'] = 1  # Buy
        self.data.loc[self.data['Returns_Lookback'] < -threshold, 'Signal'] = -1  # Sell

        # Calculate positions
        self.data['Position'] = self.data['Signal'].shift(1)

        return self.data

    def calculate_portfolio_performance(self, strategy_name):
        """Calculate portfolio performance metrics"""
        # Calculate strategy returns
        self.data['Strategy_Returns'] = self.data['Position'] * self.data['Returns']

        # Calculate cumulative returns
        self.data['Cumulative_Returns'] = (1 + self.data['Returns']).cumprod()
        self.data['Strategy_Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()

        # Calculate portfolio value
        self.data['Portfolio_Value'] = self.initial_capital * self.data['Strategy_Cumulative_Returns']

        # Calculate performance metrics
        total_return = self.data['Strategy_Cumulative_Returns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(self.data)) - 1
        volatility = self.data['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

        # Calculate maximum drawdown
        cumulative_returns = self.data['Strategy_Cumulative_Returns']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate win rate
        winning_trades = (self.data['Strategy_Returns'] > 0).sum()
        total_trades = (self.data['Position'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        performance_metrics = {
            'Strategy': strategy_name,
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': total_trades
        }

        return performance_metrics, self.data

    def plot_strategy_performance(self, strategy_data, strategy_name):
        """Plot strategy performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} Strategy Performance')

        # Price and signals
        axes[0, 0].plot(strategy_data.index, strategy_data['Close'], label='Price', alpha=0.7)
        if 'SMA_Short' in strategy_data.columns:
            axes[0, 0].plot(strategy_data.index, strategy_data['SMA_Short'], label='Short SMA', alpha=0.7)
            axes[0, 0].plot(strategy_data.index, strategy_data['SMA_Long'], label='Long SMA', alpha=0.7)
        elif 'Upper_Band' in strategy_data.columns:
            axes[0, 0].plot(strategy_data.index, strategy_data['Upper_Band'], label='Upper Band', alpha=0.7)
            axes[0, 0].plot(strategy_data.index, strategy_data['Lower_Band'], label='Lower Band', alpha=0.7)
            axes[0, 0].plot(strategy_data.index, strategy_data['SMA'], label='SMA', alpha=0.7)

        # Plot buy/sell signals
        buy_signals = strategy_data[strategy_data['Position'] == 1]
        sell_signals = strategy_data[strategy_data['Position'] == -1]

        axes[0, 0].scatter(buy_signals.index, buy_signals['Close'],
                          color='green', marker='^', s=100, label='Buy Signal')
        axes[0, 0].scatter(sell_signals.index, sell_signals['Close'],
                          color='red', marker='v', s=100, label='Sell Signal')

        axes[0, 0].set_title('Price and Trading Signals')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Cumulative returns comparison
        axes[0, 1].plot(strategy_data.index, strategy_data['Cumulative_Returns'],
                       label='Buy and Hold', alpha=0.7)
        axes[0, 1].plot(strategy_data.index, strategy_data['Strategy_Cumulative_Returns'],
                       label='Strategy', alpha=0.7)
        axes[0, 1].set_title('Cumulative Returns Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Portfolio value
        axes[1, 0].plot(strategy_data.index, strategy_data['Portfolio_Value'])
        axes[1, 0].set_title('Portfolio Value')
        axes[1, 0].set_ylabel('Portfolio Value ($)')
        axes[1, 0].grid(True)

        # Drawdown
        cumulative_returns = strategy_data['Strategy_Cumulative_Returns']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        axes[1, 1].fill_between(strategy_data.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 1].set_title('Drawdown')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def backtest_multiple_strategies(self):
        """Backtest multiple trading strategies"""
        strategies = {
            'Moving_Average_Crossover': self.moving_average_crossover,
            'Mean_Reversion': self.mean_reversion_strategy,
            'Momentum': self.momentum_strategy
        }

        results = []

        for strategy_name, strategy_func in strategies.items():
            # Apply strategy
            strategy_data = strategy_func()

            # Calculate performance
            performance_metrics, _ = self.calculate_portfolio_performance(strategy_name)
            results.append(performance_metrics)

            # Plot performance
            self.plot_strategy_performance(strategy_data, strategy_name)

        # Compare strategies
        results_df = pd.DataFrame(results)
        print("Strategy Performance Comparison:")
        print(results_df.round(4))

        return results_df

# Example usage
def algorithmic_trading_example():
    # Initialize trading strategy
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    strategy = TradingStrategy(ticker, start_date, end_date)
    strategy.fetch_data()

    # Backtest multiple strategies
    results = strategy.backtest_multiple_strategies()

    # Print summary
    print("\nBest Performing Strategy:")
    best_strategy = results.loc[results['Sharpe_Ratio'].idxmax()]
    print(f"Strategy: {best_strategy['Strategy']}")
    print(f"Sharpe Ratio: {best_strategy['Sharpe_Ratio']:.4f}")
    print(f"Annualized Return: {best_strategy['Annualized_Return']:.2%}")
    print(f"Max Drawdown: {best_strategy['Max_Drawdown']:.2%}")
```

---

## Advanced Risk Models and Financial Instruments

### GARCH and Volatility Modeling

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class GARCHModel:
    def __init__(self, returns, p=1, q=1):
        self.returns = returns
        self.p = p  # GARCH lag order
        self.q = q  # ARCH lag order
        self.fitted_model = None

    def fit_garch(self):
        """Fit GARCH(p,q) model using maximum likelihood"""
        from scipy.optimize import minimize

        # Initial parameters: [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
        initial_params = [0.01] + [0.1] * self.p + [0.8] * self.q

        # Bounds: omega > 0, alpha_i >= 0, beta_i >= 0, sum(alpha + beta) < 1
        bounds = [(1e-6, None)] + [(0, None)] * self.p + [(0, None)] * self.q

        # Constraints: sum of alpha + beta < 1
        def constraint(params):
            return 1 - sum(params[1:])

        constraints = {'type': 'ineq', 'fun': constraint}

        # Negative log-likelihood function
        def neg_log_likelihood(params):
            omega, alphas, betas = params[0], params[1:self.p+1], params[self.p+1:]

            # Initialize variance
            variance = np.zeros(len(self.returns))
            variance[0] = np.var(self.returns)

            # Calculate conditional variance
            for t in range(1, len(self.returns)):
                arch_term = sum(alphas[i] * self.returns[t-1-i]**2 for i in range(min(t, self.p)))
                garch_term = sum(betas[i] * variance[t-1-i] for i in range(min(t, self.q)))
                variance[t] = omega + arch_term + garch_term

            # Calculate log-likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + self.returns**2 / variance)
            return -log_likelihood

        # Optimize
        result = minimize(neg_log_likelihood, initial_params, bounds=bounds, constraints=constraints)

        self.fitted_model = {
            'params': result.x,
            'success': result.success,
            'log_likelihood': -result.fun
        }

        return self.fitted_model

    def forecast_volatility(self, steps=10):
        """Forecast volatility for future periods"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        params = self.fitted_model['params']
        omega, alphas, betas = params[0], params[1:self.p+1], params[self.p+1:]

        # Get last p returns and q variances
        last_returns = self.returns[-self.p:].values
        last_variances = self.get_conditional_variance()[-self.q:]

        # Forecast variance
        forecast_variance = []
        for t in range(steps):
            arch_term = sum(alphas[i] * last_returns[-(i+1)]**2 for i in range(self.p))
            garch_term = sum(betas[i] * last_variances[-(i+1)] for i in range(self.q))
            new_variance = omega + arch_term + garch_term
            forecast_variance.append(new_variance)

            # Update for next iteration
            last_returns = np.append(last_returns[1:], 0)  # Assume zero return
            last_variances = np.append(last_variances[1:], new_variance)

        return np.sqrt(forecast_variance)

    def get_conditional_variance(self):
        """Get fitted conditional variance"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        params = self.fitted_model['params']
        omega, alphas, betas = params[0], params[1:self.p+1], params[self.p+1:]

        variance = np.zeros(len(self.returns))
        variance[0] = np.var(self.returns)

        for t in range(1, len(self.returns)):
            arch_term = sum(alphas[i] * self.returns[t-1-i]**2 for i in range(min(t, self.p)))
            garch_term = sum(betas[i] * variance[t-1-i] for i in range(min(t, self.q)))
            variance[t] = omega + arch_term + garch_term

        return variance

    def plot_volatility(self):
        """Plot returns and conditional volatility"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        conditional_vol = np.sqrt(self.get_conditional_variance())

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot returns
        axes[0].plot(self.returns.index, self.returns, alpha=0.7)
        axes[0].set_title('Returns')
        axes[0].set_ylabel('Returns')
        axes[0].grid(True)

        # Plot conditional volatility
        axes[1].plot(self.returns.index, conditional_vol, color='red', alpha=0.7)
        axes[1].set_title('Conditional Volatility (GARCH)')
        axes[1].set_ylabel('Volatility')
        axes[1].set_xlabel('Date')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
def garch_example():
    # Generate sample data with volatility clustering
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Simulate GARCH(1,1) process
    omega, alpha, beta = 0.01, 0.1, 0.8
    returns = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
        returns[t] = np.random.normal(0, np.sqrt(variance[t]))

    returns_series = pd.Series(returns, index=dates)

    # Fit GARCH model
    garch_model = GARCHModel(returns_series, p=1, q=1)
    fitted_model = garch_model.fit_garch()

    print("GARCH Model Results:")
    print(f"Parameters: {fitted_model['params']}")
    print(f"Log-likelihood: {fitted_model['log_likelihood']:.4f}")

    # Forecast volatility
    forecast_vol = garch_model.forecast_volatility(steps=20)
    print(f"Volatility Forecast (next 20 days): {forecast_vol}")

    # Plot results
    garch_model.plot_volatility()
```

### Fixed Income and Bond Analytics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class BondAnalytics:
    def __init__(self, face_value, coupon_rate, maturity_years, market_rate):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.market_rate = market_rate
        self.coupon_payment = face_value * coupon_rate

    def calculate_price(self, market_rate=None):
        """Calculate bond price using present value"""
        if market_rate is None:
            market_rate = self.market_rate

        price = 0
        for t in range(1, int(self.maturity_years * 2) + 1):
            if t == int(self.maturity_years * 2):
                # Final payment includes face value
                payment = self.coupon_payment / 2 + self.face_value
            else:
                payment = self.coupon_payment / 2

            price += payment / ((1 + market_rate / 2) ** t)

        return price

    def calculate_yield_to_maturity(self, price):
        """Calculate yield to maturity using numerical methods"""
        def ytm_equation(ytm):
            return self.calculate_price(ytm) - price

        # Initial guess
        ytm_guess = self.coupon_rate

        # Solve for YTM
        ytm = fsolve(ytm_equation, ytm_guess)[0]
        return ytm

    def calculate_duration(self, market_rate=None):
        """Calculate Macaulay duration"""
        if market_rate is None:
            market_rate = self.market_rate

        price = self.calculate_price(market_rate)
        duration = 0

        for t in range(1, int(self.maturity_years * 2) + 1):
            if t == int(self.maturity_years * 2):
                payment = self.coupon_payment / 2 + self.face_value
            else:
                payment = self.coupon_payment / 2

            pv = payment / ((1 + market_rate / 2) ** t)
            duration += (t / 2) * pv / price

        return duration

    def calculate_convexity(self, market_rate=None):
        """Calculate bond convexity"""
        if market_rate is None:
            market_rate = self.market_rate

        price = self.calculate_price(market_rate)
        convexity = 0

        for t in range(1, int(self.maturity_years * 2) + 1):
            if t == int(self.maturity_years * 2):
                payment = self.coupon_payment / 2 + self.face_value
            else:
                payment = self.coupon_payment / 2

            pv = payment / ((1 + market_rate / 2) ** t)
            convexity += (t / 2) * ((t / 2) + 1) * pv / ((1 + market_rate / 2) ** 2) / price

        return convexity

    def price_sensitivity(self, market_rate_change=0.01):
        """Calculate price sensitivity to interest rate changes"""
        current_price = self.calculate_price()
        duration = self.calculate_duration()
        convexity = self.calculate_convexity()

        # First-order approximation (duration)
        price_change_duration = -duration * market_rate_change * current_price

        # Second-order approximation (convexity)
        price_change_convexity = 0.5 * convexity * (market_rate_change ** 2) * current_price

        total_price_change = price_change_duration + price_change_convexity
        new_price = current_price + total_price_change

        return {
            'Current_Price': current_price,
            'New_Price': new_price,
            'Price_Change': total_price_change,
            'Duration_Effect': price_change_duration,
            'Convexity_Effect': price_change_convexity,
            'Duration': duration,
            'Convexity': convexity
        }

    def yield_curve_analysis(self, yields, maturities):
        """Analyze yield curve and calculate spot rates"""
        # Simple bootstrapping for spot rates
        spot_rates = []

        for i, (yield_rate, maturity) in enumerate(zip(yields, maturities)):
            if i == 0:
                # First rate is the spot rate
                spot_rates.append(yield_rate)
            else:
                # Calculate spot rate using bootstrapping
                def spot_rate_equation(spot_rate):
                    price = 0
                    for j in range(i + 1):
                        if j == i:
                            payment = 100 * (1 + yields[j] / 2)  # Final payment
                        else:
                            payment = 100 * yields[j] / 2  # Coupon payment

                        if j == 0:
                            price += payment / ((1 + spot_rates[0] / 2) ** (maturity * 2))
                        else:
                            price += payment / ((1 + spot_rate / 2) ** (maturity * 2))

                    return price - 100

                spot_rate = fsolve(spot_rate_equation, yield_rate)[0]
                spot_rates.append(spot_rate)

        return spot_rates

    def plot_yield_curve(self, yields, maturities, spot_rates=None):
        """Plot yield curve and spot rate curve"""
        plt.figure(figsize=(10, 6))

        plt.plot(maturities, yields, 'o-', label='Yield to Maturity', linewidth=2)

        if spot_rates:
            plt.plot(maturities, spot_rates, 's-', label='Spot Rates', linewidth=2)

        plt.xlabel('Maturity (Years)')
        plt.ylabel('Rate (%)')
        plt.title('Yield Curve Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
def bond_analytics_example():
    # Create bond
    bond = BondAnalytics(
        face_value=1000,
        coupon_rate=0.05,  # 5% coupon
        maturity_years=10,
        market_rate=0.06   # 6% market rate
    )

    # Calculate bond metrics
    price = bond.calculate_price()
    ytm = bond.calculate_yield_to_maturity(price)
    duration = bond.calculate_duration()
    convexity = bond.calculate_convexity()

    print("Bond Analytics Results:")
    print(f"Bond Price: ${price:.2f}")
    print(f"Yield to Maturity: {ytm:.4f}")
    print(f"Macaulay Duration: {duration:.2f} years")
    print(f"Convexity: {convexity:.2f}")

    # Price sensitivity analysis
    sensitivity = bond.price_sensitivity(market_rate_change=0.01)
    print(f"\nPrice Sensitivity (1% rate increase):")
    print(f"Current Price: ${sensitivity['Current_Price']:.2f}")
    print(f"New Price: ${sensitivity['New_Price']:.2f}")
    print(f"Price Change: ${sensitivity['Price_Change']:.2f}")
    print(f"Duration Effect: ${sensitivity['Duration_Effect']:.2f}")
    print(f"Convexity Effect: ${sensitivity['Convexity_Effect']:.2f}")

    # Yield curve analysis
    yields = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
    maturities = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

    spot_rates = bond.yield_curve_analysis(yields, maturities)
    bond.plot_yield_curve(yields, maturities, spot_rates)
```

### Derivatives and Structured Products

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

class DerivativesPricer:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility

    def black_scholes_call(self):
        """Black-Scholes call option pricing"""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        call_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price

    def black_scholes_put(self):
        """Black-Scholes put option pricing"""
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return put_price

    def binary_option(self, option_type='call', payoff=1):
        """Binary (digital) option pricing"""
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

        if option_type == 'call':
            price = payoff * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # put
            price = payoff * np.exp(-self.r * self.T) * norm.cdf(-d2)

        return price

    def barrier_option(self, barrier, option_type='call', barrier_type='down_and_out'):
        """Barrier option pricing (simplified)"""
        # Simplified barrier option pricing
        if barrier_type == 'down_and_out':
            if self.S <= barrier:
                return 0
            else:
                # Adjust strike and use standard Black-Scholes
                adjusted_strike = max(self.K, barrier)
                return self.black_scholes_call() if option_type == 'call' else self.black_scholes_put()
        else:
            # Up and out
            if self.S >= barrier:
                return 0
            else:
                adjusted_strike = min(self.K, barrier)
                return self.black_scholes_call() if option_type == 'call' else self.black_scholes_put()

    def asian_option(self, n_periods=252):
        """Asian option pricing using Monte Carlo"""
        n_simulations = 10000
        payoffs = []

        for _ in range(n_simulations):
            # Generate price path
            dt = self.T / n_periods
            prices = [self.S]

            for _ in range(n_periods):
                price = prices[-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt +
                                          self.sigma * np.sqrt(dt) * np.random.normal())
                prices.append(price)

            # Calculate average price
            avg_price = np.mean(prices)

            # Calculate payoff
            if option_type == 'call':
                payoff = max(0, avg_price - self.K)
            else:
                payoff = max(0, self.K - avg_price)

            payoffs.append(payoff)

        # Discount expected payoff
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return option_price

    def implied_volatility(self, market_price, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        def objective_function(sigma):
            self.sigma = sigma
            if option_type == 'call':
                return self.black_scholes_call() - market_price
            else:
                return self.black_scholes_put() - market_price

        # Initial guess
        sigma_guess = 0.3

        # Newton-Raphson iteration
        for _ in range(100):
            f = objective_function(sigma_guess)
            if abs(f) < 1e-6:
                break

            # Calculate derivative (vega)
            d1 = (np.log(self.S / self.K) + (self.r + 0.5 * sigma_guess**2) * self.T) / (sigma_guess * np.sqrt(self.T))
            vega = self.S * np.sqrt(self.T) * norm.pdf(d1)

            sigma_guess = sigma_guess - f / vega

        return sigma_guess

class StructuredProduct:
    def __init__(self, principal, maturity, underlying_return):
        self.principal = principal
        self.maturity = maturity
        self.underlying_return = underlying_return

    def principal_protected_note(self, participation_rate=0.8, cap_rate=0.15):
        """Principal Protected Note with equity participation"""
        # Guaranteed return
        guaranteed_return = 0.02  # 2% minimum return

        # Equity participation
        equity_return = max(0, self.underlying_return) * participation_rate

        # Apply cap
        equity_return = min(equity_return, cap_rate)

        # Total return
        total_return = max(guaranteed_return, equity_return)

        return {
            'Principal': self.principal,
            'Guaranteed_Return': guaranteed_return,
            'Equity_Return': equity_return,
            'Total_Return': total_return,
            'Final_Value': self.principal * (1 + total_return)
        }

    def reverse_convertible(self, coupon_rate=0.08, knock_in_barrier=0.7):
        """Reverse Convertible Note"""
        # Regular coupon payments
        coupon_payment = self.principal * coupon_rate * self.maturity

        # Principal at risk if underlying falls below barrier
        if self.underlying_return < knock_in_barrier - 1:
            # Convert to shares at strike price
            shares_received = self.principal / (self.principal * knock_in_barrier)
            final_value = shares_received * self.principal * (1 + self.underlying_return)
        else:
            final_value = self.principal

        return {
            'Principal': self.principal,
            'Coupon_Payment': coupon_payment,
            'Final_Value': final_value,
            'Total_Return': (final_value + coupon_payment - self.principal) / self.principal
        }

    def autocallable_note(self, autocall_dates, autocall_coupons, final_coupon=0.15):
        """Autocallable Structured Note"""
        # Check if autocalled at any date
        for date, coupon in zip(autocall_dates, autocall_coupons):
            if self.underlying_return >= 0.05:  # 5% threshold
                return {
                    'Principal': self.principal,
                    'Autocall_Date': date,
                    'Coupon_Payment': self.principal * coupon,
                    'Final_Value': self.principal,
                    'Total_Return': coupon
                }

        # If not autocalled, final payoff
        if self.underlying_return >= 0:
            final_payment = self.principal * (1 + final_coupon)
        else:
            final_payment = self.principal * (1 + self.underlying_return)

        return {
            'Principal': self.principal,
            'Autocall_Date': 'Not Called',
            'Final_Value': final_payment,
            'Total_Return': (final_payment - self.principal) / self.principal
        }

# Example usage
def derivatives_example():
    # Basic option pricing
    pricer = DerivativesPricer(S=100, K=100, T=1, r=0.05, sigma=0.2)

    call_price = pricer.black_scholes_call()
    put_price = pricer.black_scholes_put()
    binary_price = pricer.binary_option('call', payoff=10)

    print("Option Pricing Results:")
    print(f"Call Option: ${call_price:.4f}")
    print(f"Put Option: ${put_price:.4f}")
    print(f"Binary Call: ${binary_price:.4f}")

    # Implied volatility
    implied_vol = pricer.implied_volatility(call_price, 'call')
    print(f"Implied Volatility: {implied_vol:.4f}")

    # Structured products
    structured = StructuredProduct(principal=10000, maturity=3, underlying_return=0.12)

    ppn = structured.principal_protected_note()
    rc = structured.reverse_convertible()

    print(f"\nStructured Products:")
    print(f"PPN Total Return: {ppn['Total_Return']:.2%}")
    print(f"Reverse Convertible Return: {rc['Total_Return']:.2%}")
```

---

## ESG Metrics and Sustainability Analysis

### ESG Scoring and Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ESGAnalyzer:
    def __init__(self):
        self.esg_data = None
        self.esg_scores = None
        self.scaler = StandardScaler()

    def generate_sample_esg_data(self, n_companies=100):
        """Generate sample ESG data for analysis"""
        np.random.seed(42)

        # Environmental metrics
        environmental_data = {
            'carbon_emissions': np.random.lognormal(10, 0.5, n_companies),  # tons CO2
            'energy_efficiency': np.random.beta(2, 5, n_companies),  # 0-1 scale
            'renewable_energy_use': np.random.beta(3, 7, n_companies),  # percentage
            'water_consumption': np.random.lognormal(8, 0.3, n_companies),  # m3
            'waste_recycling_rate': np.random.beta(4, 6, n_companies),  # percentage
            'biodiversity_impact': np.random.normal(0, 1, n_companies)  # standardized score
        }

        # Social metrics
        social_data = {
            'employee_satisfaction': np.random.beta(6, 4, n_companies),  # 0-1 scale
            'gender_diversity': np.random.beta(4, 4, n_companies),  # percentage
            'labor_rights_score': np.random.normal(0, 1, n_companies),  # standardized
            'community_investment': np.random.lognormal(6, 0.4, n_companies),  # $ millions
            'data_privacy_score': np.random.normal(0, 1, n_companies),  # standardized
            'supply_chain_ethics': np.random.beta(5, 5, n_companies)  # 0-1 scale
        }

        # Governance metrics
        governance_data = {
            'board_independence': np.random.beta(7, 3, n_companies),  # percentage
            'executive_compensation_ratio': np.random.lognormal(3, 0.5, n_companies),  # ratio
            'shareholder_rights': np.random.normal(0, 1, n_companies),  # standardized
            'transparency_score': np.random.beta(6, 4, n_companies),  # 0-1 scale
            'corruption_perception': np.random.normal(0, 1, n_companies),  # standardized
            'tax_transparency': np.random.beta(4, 6, n_companies)  # 0-1 scale
        }

        # Combine all data
        all_data = {**environmental_data, **social_data, **governance_data}
        self.esg_data = pd.DataFrame(all_data)

        # Add company names
        self.esg_data.index = [f'Company_{i+1}' for i in range(n_companies)]

        return self.esg_data

    def calculate_esg_scores(self, weights=None):
        """Calculate ESG scores for each company"""
        if self.esg_data is None:
            raise ValueError("ESG data not loaded")

        # Default weights (equal weighting)
        if weights is None:
            weights = {
                'environmental': 0.33,
                'social': 0.33,
                'governance': 0.34
            }

        # Environmental metrics (lower is better for some)
        env_metrics = ['carbon_emissions', 'water_consumption', 'executive_compensation_ratio']
        env_scores = self.esg_data.copy()

        # Invert metrics where lower is better
        for metric in env_metrics:
            if metric in env_scores.columns:
                env_scores[metric] = 1 - (env_scores[metric] - env_scores[metric].min()) / (env_scores[metric].max() - env_scores[metric].min())

        # Calculate component scores
        environmental_cols = ['carbon_emissions', 'energy_efficiency', 'renewable_energy_use',
                            'water_consumption', 'waste_recycling_rate', 'biodiversity_impact']
        social_cols = ['employee_satisfaction', 'gender_diversity', 'labor_rights_score',
                      'community_investment', 'data_privacy_score', 'supply_chain_ethics']
        governance_cols = ['board_independence', 'executive_compensation_ratio', 'shareholder_rights',
                          'transparency_score', 'corruption_perception', 'tax_transparency']

        # Calculate weighted scores
        env_score = env_scores[environmental_cols].mean(axis=1) * weights['environmental']
        social_score = self.esg_data[social_cols].mean(axis=1) * weights['social']
        gov_score = self.esg_data[governance_cols].mean(axis=1) * weights['governance']

        # Total ESG score
        total_esg_score = env_score + social_score + gov_score

        self.esg_scores = pd.DataFrame({
            'Environmental_Score': env_score,
            'Social_Score': social_score,
            'Governance_Score': gov_score,
            'Total_ESG_Score': total_esg_score
        })

        return self.esg_scores

    def esg_portfolio_optimization(self, returns_data, target_return=None, esg_constraint=0.7):
        """Optimize portfolio considering ESG scores"""
        from scipy.optimize import minimize

        # Calculate expected returns and covariance
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()

        # ESG scores
        esg_scores = self.esg_scores['Total_ESG_Score']

        n_assets = len(expected_returns)

        # Objective function: minimize portfolio variance
        def objective(weights):
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            return portfolio_var

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]

        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * expected_returns) - target_return})

        # ESG constraint: portfolio ESG score >= threshold
        constraints.append({'type': 'ineq', 'fun': lambda x: np.sum(x * esg_scores) - esg_constraint})

        # Bounds: no short selling
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial weights
        initial_weights = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x
            portfolio_esg = np.sum(optimal_weights * esg_scores)
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

            return {
                'weights': optimal_weights,
                'portfolio_esg': portfolio_esg,
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            }
        else:
            return None

    def esg_risk_analysis(self, returns_data):
        """Analyze ESG-related risks"""
        # Calculate ESG-adjusted returns
        esg_scores = self.esg_scores['Total_ESG_Score']

        # ESG risk premium (assume higher ESG scores reduce risk)
        esg_risk_premium = (esg_scores - esg_scores.mean()) * 0.01  # 1% per standard deviation

        # Adjust returns for ESG risk
        adjusted_returns = returns_data.copy()
        for col in adjusted_returns.columns:
            if col in esg_scores.index:
                adjusted_returns[col] = adjusted_returns[col] + esg_risk_premium[col]

        # Calculate ESG-adjusted Sharpe ratios
        sharpe_ratios = adjusted_returns.mean() / adjusted_returns.std()

        # ESG risk exposure
        esg_risk_exposure = pd.DataFrame({
            'ESG_Score': esg_scores,
            'Original_Sharpe': returns_data.mean() / returns_data.std(),
            'ESG_Adjusted_Sharpe': sharpe_ratios,
            'ESG_Risk_Premium': esg_risk_premium
        })

        return esg_risk_exposure

    def plot_esg_analysis(self):
        """Plot ESG analysis results"""
        if self.esg_scores is None:
            raise ValueError("ESG scores not calculated")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ESG score distribution
        axes[0, 0].hist(self.esg_scores['Total_ESG_Score'], bins=20, alpha=0.7, color='green')
        axes[0, 0].set_title('ESG Score Distribution')
        axes[0, 0].set_xlabel('ESG Score')
        axes[0, 0].set_ylabel('Frequency')

        # Component scores
        component_scores = self.esg_scores[['Environmental_Score', 'Social_Score', 'Governance_Score']]
        component_scores.boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('ESG Component Scores')
        axes[0, 1].set_ylabel('Score')

        # ESG vs performance scatter
        if hasattr(self, 'performance_data'):
            axes[1, 0].scatter(self.esg_scores['Total_ESG_Score'], self.performance_data, alpha=0.6)
            axes[1, 0].set_title('ESG Score vs Performance')
            axes[1, 0].set_xlabel('ESG Score')
            axes[1, 0].set_ylabel('Performance')

        # ESG score heatmap
        top_companies = self.esg_scores.nlargest(10, 'Total_ESG_Score')
        im = axes[1, 1].imshow(top_companies.T, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_title('Top 10 Companies ESG Scores')
        axes[1, 1].set_xticks(range(len(top_companies)))
        axes[1, 1].set_xticklabels(top_companies.index, rotation=45)
        axes[1, 1].set_yticks(range(4))
        axes[1, 1].set_yticklabels(top_companies.columns)
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

# Example usage
def esg_analysis_example():
    # Initialize ESG analyzer
    esg_analyzer = ESGAnalyzer()

    # Generate sample data
    esg_data = esg_analyzer.generate_sample_esg_data(50)

    # Calculate ESG scores
    esg_scores = esg_analyzer.calculate_esg_scores()

    print("ESG Analysis Results:")
    print(f"Average ESG Score: {esg_scores['Total_ESG_Score'].mean():.4f}")
    print(f"ESG Score Std Dev: {esg_scores['Total_ESG_Score'].std():.4f}")
    print(f"Top 5 Companies by ESG Score:")
    print(esg_scores.nlargest(5, 'Total_ESG_Score')['Total_ESG_Score'])

    # Generate sample returns for portfolio optimization
    np.random.seed(42)
    returns_data = pd.DataFrame(
        np.random.normal(0.08, 0.15, (252, 50)),
        columns=esg_data.index
    )

    # ESG portfolio optimization
    portfolio_result = esg_analyzer.esg_portfolio_optimization(
        returns_data, target_return=0.10, esg_constraint=0.7
    )

    if portfolio_result:
        print(f"\nESG Portfolio Optimization Results:")
        print(f"Portfolio ESG Score: {portfolio_result['portfolio_esg']:.4f}")
        print(f"Portfolio Return: {portfolio_result['portfolio_return']:.4f}")
        print(f"Portfolio Volatility: {portfolio_result['portfolio_volatility']:.4f}")
        print(f"Sharpe Ratio: {portfolio_result['sharpe_ratio']:.4f}")

    # ESG risk analysis
    risk_analysis = esg_analyzer.esg_risk_analysis(returns_data)
    print(f"\nESG Risk Analysis:")
    print(f"Average ESG Risk Premium: {risk_analysis['ESG_Risk_Premium'].mean():.4f}")

    # Plot results
    esg_analyzer.plot_esg_analysis()
```

### Sentiment Analysis for Financial Markets

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

class FinancialSentimentAnalyzer:
    def __init__(self):
        self.sentiment_data = None
        self.lexicon = self._create_financial_lexicon()

    def _create_financial_lexicon(self):
        """Create financial-specific sentiment lexicon"""
        financial_positive = [
            'bullish', 'rally', 'surge', 'gain', 'profit', 'growth', 'positive', 'strong',
            'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'outperform', 'favorable',
            'robust', 'solid', 'improve', 'increase', 'rise', 'climb', 'soar', 'jump'
        ]

        financial_negative = [
            'bearish', 'decline', 'drop', 'fall', 'loss', 'weak', 'negative', 'poor',
            'miss', 'disappoint', 'underperform', 'downgrade', 'sell', 'unfavorable',
            'concern', 'risk', 'worry', 'decline', 'decrease', 'plunge', 'crash', 'tumble'
        ]

        return {
            'positive': financial_positive,
            'negative': financial_negative
        }

    def generate_sample_news_data(self, n_articles=1000):
        """Generate sample financial news data"""
        np.random.seed(42)

        # Sample news headlines and content
        headlines = [
            "Company reports strong quarterly earnings",
            "Stock plunges on disappointing results",
            "Market rallies on positive economic data",
            "Investors worry about inflation concerns",
            "Tech sector shows robust growth",
            "Bank shares decline on regulatory fears",
            "Oil prices surge on supply concerns",
            "Retail sales beat expectations",
            "Fed signals potential rate cuts",
            "Earnings season off to strong start"
        ]

        # Generate articles with sentiment
        articles = []
        sentiments = []
        dates = pd.date_range('2023-01-01', periods=n_articles, freq='D')

        for i in range(n_articles):
            # Randomly select and modify headline
            base_headline = np.random.choice(headlines)

            # Add some variation
            variations = [
                f"{base_headline} - Market Update",
                f"Breaking: {base_headline}",
                f"Analysis: {base_headline}",
                f"Latest: {base_headline}",
                base_headline
            ]

            headline = np.random.choice(variations)

            # Generate content
            content = f"{headline}. " + " ".join([
                "The market reacted accordingly to the news.",
                "Analysts are closely watching the developments.",
                "Investors remain cautious about future prospects.",
                "This could have significant implications for the sector.",
                "Trading volume increased following the announcement."
            ])

            # Assign sentiment based on keywords
            sentiment = self._calculate_text_sentiment(headline + " " + content)

            articles.append({
                'date': dates[i],
                'headline': headline,
                'content': content,
                'sentiment_score': sentiment
            })
            sentiments.append(sentiment)

        self.sentiment_data = pd.DataFrame(articles)
        return self.sentiment_data

    def _calculate_text_sentiment(self, text):
        """Calculate sentiment score for text"""
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.lexicon['positive'])
        negative_count = sum(1 for word in words if word in self.lexicon['negative'])

        # Calculate sentiment score (-1 to 1)
        total_words = len(words)
        if total_words == 0:
            return 0

        sentiment_score = (positive_count - negative_count) / total_words

        # Normalize to -1 to 1 range
        return np.clip(sentiment_score * 10, -1, 1)

    def calculate_market_sentiment(self, window=7):
        """Calculate rolling market sentiment"""
        if self.sentiment_data is None:
            raise ValueError("Sentiment data not loaded")

        # Calculate rolling sentiment
        rolling_sentiment = self.sentiment_data['sentiment_score'].rolling(window=window).mean()

        # Add to sentiment data
        self.sentiment_data['rolling_sentiment'] = rolling_sentiment

        return rolling_sentiment

    def sentiment_market_correlation(self, market_returns):
        """Analyze correlation between sentiment and market returns"""
        if self.sentiment_data is None:
            raise ValueError("Sentiment data not loaded")

        # Align sentiment data with market returns
        sentiment_series = self.sentiment_data.set_index('date')['rolling_sentiment']

        # Calculate correlation
        correlation = sentiment_series.corr(market_returns)

        # Lead-lag analysis
        lags = range(-5, 6)
        correlations = []

        for lag in lags:
            if lag < 0:
                # Sentiment leads market
                corr = sentiment_series.corr(market_returns.shift(-lag))
            elif lag > 0:
                # Market leads sentiment
                corr = sentiment_series.shift(lag).corr(market_returns)
            else:
                # Same day
                corr = sentiment_series.corr(market_returns)

            correlations.append(corr)

        return {
            'contemporaneous_correlation': correlation,
            'lead_lag_correlations': dict(zip(lags, correlations)),
            'optimal_lag': lags[np.argmax(np.abs(correlations))]
        }

    def sentiment_trading_signal(self, market_returns, threshold=0.1):
        """Generate trading signals based on sentiment"""
        if self.sentiment_data is None:
            raise ValueError("Sentiment data not loaded")

        sentiment_series = self.sentiment_data.set_index('date')['rolling_sentiment']

        # Generate signals
        signals = pd.Series(0, index=sentiment_series.index)

        # Buy signal: sentiment above threshold
        signals[sentiment_series > threshold] = 1

        # Sell signal: sentiment below -threshold
        signals[sentiment_series < -threshold] = -1

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * market_returns

        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'signals': signals,
            'strategy_returns': strategy_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def plot_sentiment_analysis(self, market_returns=None):
        """Plot sentiment analysis results"""
        if self.sentiment_data is None:
            raise ValueError("Sentiment data not loaded")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Sentiment time series
        sentiment_series = self.sentiment_data.set_index('date')['rolling_sentiment']
        axes[0, 0].plot(sentiment_series.index, sentiment_series, alpha=0.7)
        axes[0, 0].set_title('Market Sentiment Over Time')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].grid(True)

        # Sentiment distribution
        axes[0, 1].hist(self.sentiment_data['sentiment_score'], bins=30, alpha=0.7)
        axes[0, 1].set_title('Sentiment Score Distribution')
        axes[0, 1].set_xlabel('Sentiment Score')
        axes[0, 1].set_ylabel('Frequency')

        # Sentiment vs market returns
        if market_returns is not None:
            sentiment_series = self.sentiment_data.set_index('date')['rolling_sentiment']
            axes[1, 0].scatter(sentiment_series, market_returns, alpha=0.6)
            axes[1, 0].set_title('Sentiment vs Market Returns')
            axes[1, 0].set_xlabel('Sentiment Score')
            axes[1, 0].set_ylabel('Market Returns')
            axes[1, 0].grid(True)

        # Sentiment heatmap
        sentiment_pivot = self.sentiment_data.set_index('date')['sentiment_score'].resample('W').mean()
        sentiment_matrix = sentiment_pivot.values.reshape(-1, 1)
        im = axes[1, 1].imshow(sentiment_matrix.T, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_title('Weekly Sentiment Heatmap')
        axes[1, 1].set_xlabel('Week')
        axes[1, 1].set_ylabel('Sentiment')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

# Example usage
def sentiment_analysis_example():
    # Initialize sentiment analyzer
    sentiment_analyzer = FinancialSentimentAnalyzer()

    # Generate sample news data
    news_data = sentiment_analyzer.generate_sample_news_data(500)

    # Calculate market sentiment
    rolling_sentiment = sentiment_analyzer.calculate_market_sentiment(window=7)

    print("Sentiment Analysis Results:")
    print(f"Average Sentiment Score: {news_data['sentiment_score'].mean():.4f}")
    print(f"Sentiment Score Std Dev: {news_data['sentiment_score'].std():.4f}")

    # Generate sample market returns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    market_returns = pd.Series(np.random.normal(0.0005, 0.02, 500), index=dates)

    # Analyze correlation
    correlation_analysis = sentiment_analyzer.sentiment_market_correlation(market_returns)
    print(f"\nSentiment-Market Correlation Analysis:")
    print(f"Contemporaneous Correlation: {correlation_analysis['contemporaneous_correlation']:.4f}")
    print(f"Optimal Lag: {correlation_analysis['optimal_lag']} days")

    # Generate trading signals
    trading_results = sentiment_analyzer.sentiment_trading_signal(market_returns, threshold=0.1)
    print(f"\nSentiment Trading Strategy Results:")
    print(f"Total Return: {trading_results['total_return']:.4f}")
    print(f"Annualized Return: {trading_results['annualized_return']:.4f}")
    print(f"Sharpe Ratio: {trading_results['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {trading_results['max_drawdown']:.4f}")

    # Plot results
    sentiment_analyzer.plot_sentiment_analysis(market_returns)
```

### Data Integration and Alternative Sources

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataIntegrator:
    def __init__(self):
        self.integrated_data = None

    def integrate_multiple_sources(self, market_data, esg_data, sentiment_data, news_data):
        """Integrate multiple data sources for comprehensive analysis"""
        # Align all data to common time index
        common_index = market_data.index.intersection(esg_data.index).intersection(sentiment_data.index)

        # Create integrated dataset
        integrated = pd.DataFrame(index=common_index)

        # Add market data
        for col in market_data.columns:
            integrated[f'market_{col}'] = market_data.loc[common_index, col]

        # Add ESG data
        for col in esg_data.columns:
            integrated[f'esg_{col}'] = esg_data.loc[common_index, col]

        # Add sentiment data
        for col in sentiment_data.columns:
            integrated[f'sentiment_{col}'] = sentiment_data.loc[common_index, col]

        # Add news data
        for col in news_data.columns:
            integrated[f'news_{col}'] = news_data.loc[common_index, col]

        self.integrated_data = integrated
        return integrated

    def create_composite_score(self, weights=None):
        """Create composite score from multiple data sources"""
        if self.integrated_data is None:
            raise ValueError("No integrated data available")

        # Default weights
        if weights is None:
            weights = {
                'market': 0.4,
                'esg': 0.3,
                'sentiment': 0.2,
                'news': 0.1
            }

        # Normalize each component
        normalized_data = pd.DataFrame(index=self.integrated_data.index)

        # Market component (returns)
        market_cols = [col for col in self.integrated_data.columns if col.startswith('market_')]
        if market_cols:
            normalized_data['market_score'] = self.integrated_data[market_cols].mean(axis=1)

        # ESG component
        esg_cols = [col for col in self.integrated_data.columns if col.startswith('esg_')]
        if esg_cols:
            normalized_data['esg_score'] = self.integrated_data[esg_cols].mean(axis=1)

        # Sentiment component
        sentiment_cols = [col for col in self.integrated_data.columns if col.startswith('sentiment_')]
        if sentiment_cols:
            normalized_data['sentiment_score'] = self.integrated_data[sentiment_cols].mean(axis=1)

        # News component
        news_cols = [col for col in self.integrated_data.columns if col.startswith('news_')]
        if news_cols:
            normalized_data['news_score'] = self.integrated_data[news_cols].mean(axis=1)

        # Calculate composite score
        composite_score = (
            weights['market'] * normalized_data['market_score'] +
            weights['esg'] * normalized_data['esg_score'] +
            weights['sentiment'] * normalized_data['sentiment_score'] +
            weights['news'] * normalized_data['news_score']
        )

        return composite_score

    def analyze_data_quality(self):
        """Analyze data quality and completeness"""
        if self.integrated_data is None:
            raise ValueError("No integrated data available")

        # Missing data analysis
        missing_data = self.integrated_data.isnull().sum()
        missing_percentage = (missing_data / len(self.integrated_data)) * 100

        # Data coverage
        coverage = (len(self.integrated_data) - missing_data) / len(self.integrated_data) * 100

        # Data quality metrics
        quality_metrics = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percentage,
            'Coverage_Percentage': coverage
        })

        return quality_metrics

    def plot_integrated_analysis(self):
        """Plot integrated data analysis"""
        if self.integrated_data is None:
            raise ValueError("No integrated data available")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Data coverage over time
        coverage = self.integrated_data.notna().sum(axis=1)
        axes[0, 0].plot(coverage.index, coverage, alpha=0.7)
        axes[0, 0].set_title('Data Coverage Over Time')
        axes[0, 0].set_ylabel('Number of Available Variables')
        axes[0, 0].grid(True)

        # Correlation heatmap
        correlation_matrix = self.integrated_data.corr()
        im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[0, 1].set_title('Variable Correlation Matrix')
        plt.colorbar(im, ax=axes[0, 1])

        # Data source distribution
        source_counts = {}
        for col in self.integrated_data.columns:
            source = col.split('_')[0]
            source_counts[source] = source_counts.get(source, 0) + 1

        axes[1, 0].pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%')
        axes[1, 0].set_title('Data Source Distribution')

        # Time series of key metrics
        if 'market_returns' in self.integrated_data.columns:
            axes[1, 1].plot(self.integrated_data.index, self.integrated_data['market_returns'],
                           alpha=0.7, label='Market Returns')
        if 'sentiment_score' in self.integrated_data.columns:
            axes[1, 1].plot(self.integrated_data.index, self.integrated_data['sentiment_score'],
                           alpha=0.7, label='Sentiment Score')

        axes[1, 1].set_title('Key Metrics Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
def data_integration_example():
    # Initialize data integrator
    integrator = DataIntegrator()

    # Generate sample data from different sources
    dates = pd.date_range('2023-01-01', periods=252, freq='D')

    # Market data
    market_data = pd.DataFrame({
        'returns': np.random.normal(0.0005, 0.02, 252),
        'volume': np.random.lognormal(10, 0.5, 252)
    }, index=dates)

    # ESG data
    esg_data = pd.DataFrame({
        'esg_score': np.random.beta(3, 3, 252),
        'carbon_emissions': np.random.lognormal(8, 0.3, 252)
    }, index=dates)

    # Sentiment data
    sentiment_data = pd.DataFrame({
        'sentiment_score': np.random.normal(0, 0.5, 252),
        'news_volume': np.random.poisson(50, 252)
    }, index=dates)

    # News data
    news_data = pd.DataFrame({
        'news_sentiment': np.random.normal(0, 0.3, 252),
        'article_count': np.random.poisson(100, 252)
    }, index=dates)

    # Integrate data
    integrated_data = integrator.integrate_multiple_sources(
        market_data, esg_data, sentiment_data, news_data
    )

    print("Data Integration Results:")
    print(f"Integrated dataset shape: {integrated_data.shape}")
    print(f"Data sources: {[col.split('_')[0] for col in integrated_data.columns]}")

    # Create composite score
    composite_score = integrator.create_composite_score()
    print(f"Composite Score Statistics:")
    print(f"Mean: {composite_score.mean():.4f}")
    print(f"Std Dev: {composite_score.std():.4f}")

    # Analyze data quality
    quality_metrics = integrator.analyze_data_quality()
    print(f"\nData Quality Analysis:")
    print(quality_metrics)

    # Plot integrated analysis
    integrator.plot_integrated_analysis()
```

---

## References and Further Reading

_This framework represents a synthesis of concepts from various sources including academic textbooks, research papers, industry best practices, and practical experience. For specific technical details and formal definitions, please refer to the original sources and current literature in each field._

---

_Last Updated: [Current Date]_
_Version: 1.0_

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

def calculate_p_value_prediction_accuracy(correct_predictions, total_predictions, expected_accuracy=0.5):
    """
    Calculate p-value for prediction model accuracy test
    """
    observed_accuracy = correct_predictions / total_predictions
    se = np.sqrt(expected_accuracy * (1 - expected_accuracy) / total_predictions)
    z_score = (observed_accuracy - expected_accuracy) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return p_value, z_score

def plot_p_value_distribution():
    """Visualize p-value distribution under null hypothesis for prediction accuracy"""
    # Generate many p-values under null hypothesis
    np.random.seed(42)
    n_simulations = 10000
    p_values = []

    for _ in range(n_simulations):
        # Simulate prediction accuracy experiment under null hypothesis
        correct_predictions = np.random.binomial(100, 0.5)
        p_val, _ = calculate_p_value_prediction_accuracy(correct_predictions, 100)
        p_values.append(p_val)

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(p_values, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    plt.xlabel('P-Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('P-Value Distribution Under Null Hypothesis (Random Prediction Model)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('./visualizations/p_value_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Check uniform distribution
    print(f"Mean p-value: {np.mean(p_values):.3f}")
    print(f"Proportion < 0.05: {np.mean(np.array(p_values) < 0.05):.3f}")

def analyze_quarter_end_predictions():
    """
    Analyze whether a prediction model's quarter-end accuracy is better than chance
    """
    # Simulate quarter-end prediction data
    np.random.seed(42)
    quarters = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 
                'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024']
    
    # Simulate model predictions (1 = correct, 0 = incorrect)
    # Under null hypothesis: model is random (25% accuracy for 4 possible outcomes)
    random_predictions = np.random.binomial(1, 0.25, len(quarters))
    
    # Simulate skilled model predictions (60% accuracy)
    skilled_predictions = np.random.binomial(1, 0.60, len(quarters))
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Quarter': quarters,
        'Random_Model': random_predictions,
        'Skilled_Model': skilled_predictions
    })
    
    # Calculate p-values for both models
    random_accuracy = np.sum(random_predictions) / len(quarters)
    skilled_accuracy = np.sum(skilled_predictions) / len(quarters)
    
    # Test against null hypothesis of 25% accuracy (random guessing)
    random_p_val, random_z = calculate_p_value_prediction_accuracy(
        np.sum(random_predictions), len(quarters), expected_accuracy=0.25
    )
    
    skilled_p_val, skilled_z = calculate_p_value_prediction_accuracy(
        np.sum(skilled_predictions), len(quarters), expected_accuracy=0.25
    )
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy comparison
    models = ['Random Model', 'Skilled Model']
    accuracies = [random_accuracy, skilled_accuracy]
    p_values = [random_p_val, skilled_p_val]
    
    bars = ax1.bar(models, accuracies, color=['red', 'green'], alpha=0.7)
    ax1.axhline(y=0.25, color='black', linestyle='--', linewidth=2, label='Random Chance (25%)')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy vs Random Chance', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 0.8)
    
    # Add p-value annotations
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'p={p_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot quarterly results
    quarters_short = [q.split('_')[0] for q in quarters]
    ax2.plot(quarters_short, np.cumsum(random_predictions), 
             marker='o', label='Random Model', color='red', linewidth=2, markersize=8)
    ax2.plot(quarters_short, np.cumsum(skilled_predictions), 
             marker='s', label='Skilled Model', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Quarter', fontsize=12)
    ax2.set_ylabel('Cumulative Correct Predictions', fontsize=12)
    ax2.set_title('Cumulative Performance Over Time', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./visualizations/quarter_end_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

if __name__ == "__main__":
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('./visualizations', exist_ok=True)
    
    print("Generating P-Value Distribution Plot...")
    plot_p_value_distribution()
    
    print("\nGenerating Quarter-End Predictions Analysis...")
    quarter_results = analyze_quarter_end_predictions()
    
    print("\nVisualizations saved to ./visualizations/")
    print("- p_value_distribution.png")
    print("- quarter_end_predictions.png")

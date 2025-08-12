"""
Professional Visualizations for Skewness and Kurtosis Examples
Author: Oliver Lei
Date: August 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def calculate_mode(data, bins=30):
    """Calculate mode using histogram method for continuous data"""
    hist, bin_edges = np.histogram(data, bins=bins)
    mode_idx = np.argmax(hist)
    mode_value = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    return mode_value

def create_skewness_examples():
    """Create examples of different types of skewness"""
    
    # Generate data for different skewness types
    np.random.seed(42)
    
    # Negative skewness (left-skewed)
    negative_skew = stats.skewnorm.rvs(a=-3, loc=5, scale=1, size=1000)
    
    # Positive skewness (right-skewed)
    positive_skew = stats.skewnorm.rvs(a=3, loc=2, scale=1, size=1000)
    
    # Zero skewness (symmetric)
    zero_skew = np.random.normal(loc=5, scale=1, size=1000)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot negative skewness
    axes[0].hist(negative_skew, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(np.mean(negative_skew), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(negative_skew):.2f}')
    axes[0].axvline(np.median(negative_skew), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(negative_skew):.2f}')
    axes[0].axvline(calculate_mode(negative_skew), color='orange', linestyle='--', linewidth=2, label=f'Mode: {calculate_mode(negative_skew):.2f}')
    axes[0].set_title('Negative Skewness (Left-Skewed)\nMean < Median < Mode', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Values')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot positive skewness
    axes[1].hist(positive_skew, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].axvline(np.mean(positive_skew), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(positive_skew):.2f}')
    axes[1].axvline(np.median(positive_skew), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(positive_skew):.2f}')
    axes[1].axvline(calculate_mode(positive_skew), color='orange', linestyle='--', linewidth=2, label=f'Mode: {calculate_mode(positive_skew):.2f}')
    axes[1].set_title('Positive Skewness (Right-Skewed)\nMode < Median < Mean', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Values')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot zero skewness
    axes[2].hist(zero_skew, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2].axvline(np.mean(zero_skew), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(zero_skew):.2f}')
    axes[2].axvline(np.median(zero_skew), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(zero_skew):.2f}')
    axes[2].axvline(calculate_mode(zero_skew), color='orange', linestyle='--', linewidth=2, label=f'Mode: {calculate_mode(zero_skew):.2f}')
    axes[2].set_title('Zero Skewness (Symmetric)\nMean = Median = Mode', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Values')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/skewness_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_kurtosis_examples():
    """Create examples of different types of kurtosis"""
    
    # Generate data for different kurtosis types
    np.random.seed(42)
    
    # High kurtosis (leptokurtic) - Student's t with low degrees of freedom
    high_kurtosis = np.random.standard_t(df=3, size=1000)
    
    # Low kurtosis (platykurtic) - Uniform distribution
    low_kurtosis = np.random.uniform(-3, 3, size=1000)
    
    # Normal kurtosis (mesokurtic) - Standard normal
    normal_kurtosis = np.random.standard_normal(size=1000)
    
    # Create figure with better layout to show tail differences
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Top row: Full distributions
    # Plot high kurtosis
    axes[0, 0].hist(high_kurtosis, bins=50, alpha=0.7, color='purple', edgecolor='black', density=True)
    x_range = np.linspace(high_kurtosis.min(), high_kurtosis.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(high_kurtosis), np.std(high_kurtosis))
    axes[0, 0].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal')
    
    # Calculate kurtosis statistics
    excess_kurtosis = stats.kurtosis(high_kurtosis)  # scipy returns excess kurtosis by default
    kurtosis_val = excess_kurtosis + 3  # Full kurtosis = excess + 3
    
    axes[0, 0].set_title(f'High Kurtosis (Leptokurtic)\nExcess Kurtosis: {excess_kurtosis:.2f}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Values')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].text(0.02, 0.9, '• More extreme events\n• Higher risk of outliers\n• Like cryptocurrency returns', 
                transform=axes[0, 0].transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7))
    axes[0, 0].legend(loc='lower left', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot low kurtosis
    axes[0, 1].hist(low_kurtosis, bins=50, alpha=0.7, color='orange', edgecolor='black', density=True)
    x_range = np.linspace(low_kurtosis.min(), low_kurtosis.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(low_kurtosis), np.std(low_kurtosis))
    axes[0, 1].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal')
    
    # Calculate kurtosis statistics
    excess_kurtosis = stats.kurtosis(low_kurtosis)  # scipy returns excess kurtosis by default
    kurtosis_val = excess_kurtosis + 3  # Full kurtosis = excess + 3
    
    axes[0, 1].set_title(f'Low Kurtosis (Platykurtic)\nExcess Kurtosis: {excess_kurtosis:.2f}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].text(0.02, 0.9, '• Fewer extreme events\n• More predictable outcomes\n• Like government bonds', 
                transform=axes[0, 1].transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="peachpuff", alpha=0.7))
    axes[0, 1].legend(loc='lower left', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot normal kurtosis
    axes[0, 2].hist(normal_kurtosis, bins=50, alpha=0.7, color='teal', edgecolor='black', density=True)
    x_range = np.linspace(normal_kurtosis.min(), normal_kurtosis.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(normal_kurtosis), np.std(normal_kurtosis))
    axes[0, 2].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal')
    
    # Calculate kurtosis statistics
    excess_kurtosis = stats.kurtosis(normal_kurtosis)  # scipy returns excess kurtosis by default
    kurtosis_val = excess_kurtosis + 3  # Full kurtosis = excess + 3
    
    axes[0, 2].set_title(f'Normal Kurtosis (Mesokurtic)\nExcess Kurtosis: {excess_kurtosis:.2f}', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Values')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].text(0.02, 0.9, '• Standard bell curve\n• Moderate extremes\n• Baseline for comparison', 
                transform=axes[0, 2].transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7))
    axes[0, 2].legend(loc='lower left', fontsize=9)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Bottom row: Zoom in on tails to show differences clearly
    # High kurtosis tails
    axes[1, 0].hist(high_kurtosis, bins=100, alpha=0.7, color='purple', edgecolor='black', density=True)
    x_range = np.linspace(high_kurtosis.min(), high_kurtosis.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(high_kurtosis), np.std(high_kurtosis))
    axes[1, 0].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal')
    axes[1, 0].set_xlim(-5, 5)  # Focus on tail region
    axes[1, 0].set_title('High Kurtosis: Fat Tails\n(More extreme values than normal)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Values')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].text(0.02, 0.9, 'FAT TAILS:\n• Higher density in tails\n• More extreme events\n• Higher risk', 
                transform=axes[1, 0].transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    axes[1, 0].legend(loc='lower left', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Low kurtosis tails
    axes[1, 1].hist(low_kurtosis, bins=100, alpha=0.7, color='orange', edgecolor='black', density=True)
    x_range = np.linspace(low_kurtosis.min(), low_kurtosis.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(low_kurtosis), np.std(low_kurtosis))
    axes[1, 1].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal')
    axes[1, 1].set_xlim(-3, 3)  # Focus on tail region
    axes[1, 1].set_title('Low Kurtosis: Thin Tails\n(Fewer extreme values than normal)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Values')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].text(0.02, 0.9, 'THIN TAILS:\n• Lower density in tails\n• Fewer extreme events\n• Lower risk', 
                transform=axes[1, 1].transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
    axes[1, 1].legend(loc='lower left', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Normal kurtosis for comparison
    axes[1, 2].hist(normal_kurtosis, bins=100, alpha=0.7, color='teal', edgecolor='black', density=True)
    x_range = np.linspace(normal_kurtosis.min(), normal_kurtosis.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(normal_kurtosis), np.std(normal_kurtosis))
    axes[1, 2].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal')
    axes[1, 2].set_xlim(-4, 4)  # Focus on tail region
    axes[1, 2].set_title('Normal Kurtosis: Standard Tails\n(Baseline for comparison)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Values')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].text(0.02, 0.9, 'STANDARD TAILS:\n• Normal density in tails\n• Expected extremes\n• Baseline risk', 
                transform=axes[1, 2].transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
    axes[1, 2].legend(loc='lower left', fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/kurtosis_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_combined_comparison():
    """Create a comprehensive comparison of skewness and kurtosis"""
    
    np.random.seed(42)
    
    # Generate different distributions
    normal = np.random.normal(0, 1, 1000)
    lognormal = np.random.lognormal(0, 1, 1000)
    uniform = np.random.uniform(-2, 2, 1000)
    t_dist = np.random.standard_t(df=3, size=1000)
    
    # Calculate statistics
    distributions = {
        'Normal': normal,
        'Log-normal (Right-skewed)': lognormal,
        'Uniform (Low kurtosis)': uniform,
        'Student-t (High kurtosis)': t_dist
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
    
    for i, (name, data) in enumerate(distributions.items()):
        axes[i].hist(data, bins=30, alpha=0.7, color=colors[i], edgecolor='black', density=True)
        
        # Add normal distribution for comparison
        x_range = np.linspace(data.min(), data.max(), 100)
        normal_pdf = stats.norm.pdf(x_range, np.mean(data), np.std(data))
        axes[i].plot(x_range, normal_pdf, 'r--', linewidth=2, label='Normal Distribution')
        
        # Calculate and display statistics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        axes[i].set_title(f'{name}\nSkewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_financial_example():
    """Create a financial example showing skewness and kurtosis in returns"""
    
    np.random.seed(42)
    
    # Simulate financial returns with different characteristics
    days = 252 * 5  # 5 years of daily data
    
    # Normal returns (low kurtosis, zero skewness) - like government bonds
    normal_returns = np.random.normal(0.0005, 0.02, days)
    
    # Skewed returns (positive skewness, high kurtosis) - like growth stocks
    skewed_returns = stats.skewnorm.rvs(a=2, loc=0.0005, scale=0.02, size=days)
    
    # High kurtosis returns (fat tails) - like cryptocurrency
    high_kurt_returns = np.random.standard_t(df=3, size=days) * 0.02 + 0.0005
    
    # Create cumulative returns
    normal_cumulative = np.cumprod(1 + normal_returns)
    skewed_cumulative = np.cumprod(1 + skewed_returns)
    high_kurt_cumulative = np.cumprod(1 + high_kurt_returns)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot return distributions with explanations
    axes[0, 0].hist(normal_returns, bins=50, alpha=0.7, color='blue', density=True, label='Normal')
    axes[0, 0].set_title('Normal Returns Distribution\n(Like Government Bonds)', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Daily Returns')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].text(0.02, 0.9, '• Symmetric distribution\n• Predictable outcomes\n• Low risk of extremes', 
                    transform=axes[0, 0].transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(skewed_returns, bins=50, alpha=0.7, color='red', density=True, label='Skewed')
    axes[0, 1].set_title('Skewed Returns Distribution\n(Like Growth Stocks)', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Daily Returns')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].text(0.02, 0.9, '• Asymmetric distribution\n• More frequent small losses\n• Occasional large gains', 
                    transform=axes[0, 1].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot cumulative returns with explanations and better scaling
    axes[1, 0].plot(normal_cumulative, label='Normal (Bonds)', linewidth=3, color='blue', alpha=0.8)
    axes[1, 0].plot(skewed_cumulative, label='Skewed (Growth)', linewidth=3, color='red', alpha=0.8)
    axes[1, 0].plot(high_kurt_cumulative, label='High Kurtosis (Crypto)', linewidth=3, color='purple', alpha=0.8)
    axes[1, 0].set_title('Cumulative Returns Comparison\n($1 Investment Over 5 Years)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Trading Days')
    axes[1, 0].set_ylabel('Cumulative Return ($)')
    
    # Add final values as text annotations
    final_normal = normal_cumulative[-1]
    final_skewed = skewed_cumulative[-1]
    final_high_kurt = high_kurt_cumulative[-1]
    
    axes[1, 0].text(0.02, 0.9, f'Final Values:\n• Bonds: ${final_normal:.2f}\n• Growth: ${final_skewed:.2f}\n• Crypto: ${final_high_kurt:.2f}', 
                    transform=axes[1, 0].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Set y-axis limits to show differences clearly
    y_min = min(normal_cumulative.min(), skewed_cumulative.min(), high_kurt_cumulative.min())
    y_max = max(normal_cumulative.max(), skewed_cumulative.max(), high_kurt_cumulative.max())
    axes[1, 0].set_ylim(y_min * 0.95, y_max * 1.05)
    
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot QQ plots to show deviation from normality
    stats.probplot(normal_returns, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Normal vs Theoretical Normal\n(Shows how well data fits normal distribution)', fontweight='bold', fontsize=12)
    axes[1, 1].text(0.02, 0.9, '• Points on line = normal\n• Deviations = non-normal\n• Tails show extremes', 
                    transform=axes[1, 1].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/financial_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("Generating skewness examples...")
    create_skewness_examples()
    
    print("Generating kurtosis examples...")
    create_kurtosis_examples()
    
    print("Generating combined comparison...")
    create_combined_comparison()
    
    print("Generating financial example...")
    create_financial_example()
    
    print("All visualizations saved to 'visualizations/' directory!")

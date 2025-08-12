"""
Test script to verify mean, median, and mode calculations for skewness examples
"""

import numpy as np
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate the same data as in the visualization
negative_skew = stats.skewnorm.rvs(a=-3, loc=5, scale=1, size=1000)
positive_skew = stats.skewnorm.rvs(a=3, loc=2, scale=1, size=1000)
zero_skew = np.random.normal(loc=5, scale=1, size=1000)

print("=== NEGATIVE SKEWNESS (Left-Skewed) ===")
print(f"Mean: {np.mean(negative_skew):.3f}")
print(f"Median: {np.median(negative_skew):.3f}")
print(f"Mode (scipy): {stats.mode(negative_skew)[0]:.3f}")
print(f"Expected relationship: Mean < Median < Mode")
print(f"Actual relationship: {np.mean(negative_skew):.3f} < {np.median(negative_skew):.3f} < {stats.mode(negative_skew)[0]:.3f}")
print(f"Skewness: {stats.skew(negative_skew):.3f}")
print()

print("=== POSITIVE SKEWNESS (Right-Skewed) ===")
print(f"Mean: {np.mean(positive_skew):.3f}")
print(f"Median: {np.median(positive_skew):.3f}")
print(f"Mode (scipy): {stats.mode(positive_skew)[0]:.3f}")
print(f"Expected relationship: Mode < Median < Mean")
print(f"Actual relationship: {stats.mode(positive_skew)[0]:.3f} < {np.median(positive_skew):.3f} < {np.mean(positive_skew):.3f}")
print(f"Skewness: {stats.skew(positive_skew):.3f}")
print()

print("=== ZERO SKEWNESS (Symmetric) ===")
print(f"Mean: {np.mean(zero_skew):.3f}")
print(f"Median: {np.median(zero_skew):.3f}")
print(f"Mode (scipy): {stats.mode(zero_skew)[0]:.3f}")
print(f"Expected relationship: Mean ≈ Median ≈ Mode")
print(f"Skewness: {stats.skew(zero_skew):.3f}")
print()

# Test a better mode calculation method
def calculate_mode(data, bins=30):
    """Calculate mode using histogram method"""
    hist, bin_edges = np.histogram(data, bins=bins)
    mode_idx = np.argmax(hist)
    mode_value = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    return mode_value

print("=== IMPROVED MODE CALCULATION ===")
print(f"Negative skew mode (histogram): {calculate_mode(negative_skew):.3f}")
print(f"Positive skew mode (histogram): {calculate_mode(positive_skew):.3f}")
print(f"Zero skew mode (histogram): {calculate_mode(zero_skew):.3f}")

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from regression_module_v2 import RegressionPipeline
import matplotlib.pyplot as plt
import numpy as np

# Load Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test all models
model = RegressionPipeline()
results_all = model.test_all_models(X_train, X_test, y_train, y_test)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: R² Score Comparison
models = list(results_all.keys())
r2_scores = [results_all[m]['r2'] for m in models]

axes[0, 0].bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0, 0].set_ylabel('R² Score', fontsize=11)
axes[0, 0].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim([0, 1.0])
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Add value labels
for i, (model, r2) in enumerate(zip(models, r2_scores)):
    axes[0, 0].text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 2: RMSE Comparison
rmse_scores = [results_all[m]['rmse'] for m in models]

axes[0, 1].bar(models, rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0, 1].set_ylabel('RMSE', fontsize=11)
axes[0, 1].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Add value labels
for i, (model, rmse) in enumerate(zip(models, rmse_scores)):
    axes[0, 1].text(i, rmse + 1, f'{rmse:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 3: MAE Comparison
mae_scores = [results_all[m]['mae'] for m in models]

axes[1, 0].bar(models, mae_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[1, 0].set_ylabel('MAE', fontsize=11)
axes[1, 0].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# Add value labels
for i, (model, mae) in enumerate(zip(models, mae_scores)):
    axes[1, 0].text(i, mae + 0.5, f'{mae:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 4: Dataset Info
axes[1, 1].axis('off')
dataset_info = f"""
DATASET INFORMATION
{'='*40}

Samples: {X.shape[0]}
Features: {X.shape[1]}
Train/Test Split: {len(X_train)}/{len(X_test)}

TASK: Medical Regression
Predict: Disease Progression (y)

BEST MODEL: Gradient Boosting
R² Score: {results_all['Gradient Boosting']['r2']:.4f}
RMSE: {results_all['Gradient Boosting']['rmse']:.2f}
MAE: {results_all['Gradient Boosting']['mae']:.2f}
"""

axes[1, 1].text(0.05, 0.5, dataset_info, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

plt.tight_layout()
plt.savefig('regression_diabetes_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as: regression_diabetes_results.png")
plt.show()

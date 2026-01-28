from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from classification_module_v2 import ClassificationPipeline
import matplotlib.pyplot as plt
import numpy as np

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test all models
model = ClassificationPipeline()
results_all = model.test_all_models(
    X_train, X_test, y_train, y_test, 
    target_names=cancer.target_names
)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model Comparison (Accuracy)
models = list(results_all.keys())
accuracies = list(results_all.values())

axes[0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_title('Classification Model Comparison\n(Breast Cancer Dataset)', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.8, 1.0])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (model, acc) in enumerate(zip(models, accuracies)):
    axes[0].text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# Rotate x labels
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Dataset Info
dataset_info = [
    f"Samples: {X.shape[0]}",
    f"Features: {X.shape[1]}",
    f"Classes: {len(np.unique(y))}",
    f"Train/Test: {len(X_train)}/{len(X_test)}"
]

axes[1].axis('off')
info_text = "DATASET INFORMATION\n" + "=" * 30 + "\n"
info_text += "\n".join(dataset_info)
info_text += "\n\nTASK: Medical Diagnosis\nPredict: Malignant vs Benign"

axes[1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('classification_breast_cancer_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as: classification_breast_cancer_results.png")
plt.show()

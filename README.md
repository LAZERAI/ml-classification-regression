# ML Classification & Regression Pipeline

Generic machine learning pipelines for classification and regression tasks with automatic model comparison and visualization.

## Overview

Two reusable pipeline classes that standardize ML workflows:

- **ClassificationPipeline** - Train and compare 4 classification models
- **RegressionPipeline** - Train and compare 4 regression models

Both handle data preprocessing, cross-validation, and performance evaluation automatically.

## Models

**Classification:** Logistic Regression, Decision Tree, Random Forest, SVM

**Regression:** Linear, Gradient Boosting, Random Forest, SVR

## Quick Example

```python
from classification_module_v2 import ClassificationPipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compare all models
model = ClassificationPipeline()
results = model.test_all_models(X_train, X_test, y_train, y_test)

# Visualize
model.plot_model_accuracies(results)
```

## Results

**Classification (Diabetes):** Random Forest 75.76%, SVM 74.46%, Logistic 73.59%, Decision Tree 71.86%

**Regression (Car Price):** Random Forest 99.64% R², Gradient Boosting 99.51%, SVR 99.49%, Linear 98.69%

## Features

✓ No data leakage - StandardScaler fitted only on training data  
✓ 5-fold cross-validation for honest evaluation  
✓ Compare all models with one method call  
✓ Built-in visualization and metrics  
✓ Reusable with any dataset

## Requirements

```
scikit-learn pandas numpy matplotlib
```

## Install

```bash
pip install scikit-learn pandas numpy matplotlib
```

## Files

- `classification_module_v2.py` - Classification pipeline class
- `regression_module_v2.py` - Regression pipeline class  
- `simple_classification_example.py` - Demo classification task
- `simple_regression_example.py` - Demo regression task

## How to Use with Your Data

1. Prepare a CSV with features and target column
2. Load and split: `train_test_split(X, y, test_size=0.3)`
3. Create pipeline: `model = ClassificationPipeline()`
4. Test models: `results = model.test_all_models(...)`
5. Visualize: `model.plot_model_accuracies(results)`

See example scripts for complete code.

## License

MIT

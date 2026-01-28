# Quick Start Guide - Classification & Regression Modules

## Running the Examples

### Classification Example (Iris Dataset)
```bash
cd c:\Users\Lazerai\Documents\ml-pipeline-iris-v2
python example_iris_classification.py
```

**What it shows:**
- How to load Iris dataset
- Training all 4 classification models
- Comparing model accuracy
- Making predictions
- Understanding confusion matrix

**Expected output:** ~30 seconds, shows metrics for 4 models

### Regression Example (Iris Dataset)
```bash
cd c:\Users\Lazerai\Documents\ml-pipeline-iris-v2
python example_iris_regression.py
```

**What it shows:**
- How to use Iris for regression (predict petal length)
- Training all 5 regression models
- Comparing R², RMSE, MAE scores
- Error analysis
- Model performance comparison

**Expected output:** ~30 seconds, shows metrics for 5 models

---

## Minimal Usage Example

### Classification
```python
from classification_module_v2 import ClassificationPipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# Create and train
clf = ClassificationPipeline('Random Forest')
clf.train(X_train, y_train)

# Evaluate
results = clf.evaluate(X_test, y_test, target_names=iris.target_names)

# Predict
predictions = clf.predict(X_test)
```

### Regression
```python
from regression_module_v2 import RegressionPipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Prepare data (predict petal length)
iris = load_iris()
X = iris.data[:, [0,1,3]]  # Exclude petal length
y = iris.data[:, 2]        # Petal length
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train
reg = RegressionPipeline('Random Forest')
reg.train(X_train, y_train)

# Evaluate
results = reg.evaluate(X_test, y_test)

# Predict
predictions = reg.predict(X_test)
```

---

## Available Models

### Classification Models
1. **Logistic Regression** - Linear classifier, fast
2. **Decision Tree** - Simple tree-based classifier
3. **Random Forest** - Ensemble of trees, robust
4. **SVM** - Complex boundaries, good for difficult patterns

### Regression Models
1. **Linear** - Straight line fit, simplest
2. **Ridge** - Linear with L2 regularization
3. **Lasso** - Linear with L1 regularization (feature selection)
4. **Random Forest** - Ensemble trees for regression
5. **SVR** - Support Vector Regressor, complex boundaries

---

## Files in This Project

| File | Purpose |
|------|---------|
| `classification_module_v2.py` | ClassificationPipeline class |
| `regression_module_v2.py` | RegressionPipeline class |
| `DETAILED_EXPLANATION.md` | Complete line-by-line explanation |
| `example_iris_classification.py` | Classification example |
| `example_iris_regression.py` | Regression example |
| `QUICK_START.md` | This file |

---

## Key Parameters Reference

### Classification: Logistic Regression
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- `max_iter=1000`: Training iterations (higher = more thorough)
- `random_state=42`: Reproducibility seed

### Classification: Decision Tree
```python
DecisionTreeClassifier(max_depth=3, random_state=42)
```
- `max_depth=3`: Tree depth (higher = more complex, risk of overfitting)
- `random_state=42`: Reproducibility

### Classification: Random Forest
```python
RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
```
- `n_estimators=100`: Number of trees (more = better but slower)
- `max_depth=3`: Each tree's depth
- `random_state=42`: Reproducibility

### Classification: SVM
```python
SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
```
- `kernel='rbf'`: Boundary type (rbf = curved, linear = straight)
- `C=1.0`: Regularization (higher = stricter, lower = lenient)
- `gamma='scale'`: Influence range (auto-calculated)
- `probability=True`: Enable probability estimates
- `random_state=42`: Reproducibility

### Regression: Ridge
```python
Ridge(alpha=1.0, random_state=42)
```
- `alpha=1.0`: Regularization strength (higher = simpler model)

### Regression: Lasso
```python
Lasso(alpha=0.1, random_state=42)
```
- `alpha=0.1`: More aggressive than Ridge

### Regression: Random Forest
```python
RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
```
- `max_depth=10`: Deeper than classification (regression needs complexity)

### Regression: SVR
```python
SVR(kernel='rbf', C=100, gamma='scale')
```
- `C=100`: Higher than SVC for regression

---

## Understanding Outputs

### Classification Results
```python
results = {
    'Accuracy': 0.9333,           # % correct predictions
    'Predictions': array([...]),   # Predicted labels
    'Classification Report': "...", # Precision, recall, F1
    'Confusion Matrix': array([...])  # Prediction breakdown
}
```

### Regression Results
```python
results = {
    'MSE': 0.0456,                # Mean squared error
    'RMSE': 0.2135,               # Root mean squared error
    'MAE': 0.1567,                # Mean absolute error
    'R2 Score': 0.8934,           # Variance explained
    'Predictions': array([...])    # Predicted values
}
```

---

## Evaluation Metrics Cheat Sheet

### Classification
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Overall correctness
- **Precision**: TP/(TP+FP) - Of predicted positive, how many correct?
- **Recall**: TP/(TP+FN) - Of actual positive, how many found?
- **F1-Score**: 2×(Precision×Recall)/(Precision+Recall) - Balance of both

### Regression
- **R² Score**: Higher is better (0 to 1), % variance explained
- **RMSE**: Lower is better, penalizes large errors
- **MAE**: Lower is better, average error magnitude
- **MSE**: Lower is better, squared errors (RMSE = √MSE)

---

## Common Modifications

### Try a Different Model
```python
clf = ClassificationPipeline('SVM')  # Instead of 'Random Forest'
```

### Adjust Cross-Validation Folds
```python
cv_score = clf.train(X_train, y_train, cv=10)  # Instead of cv=5
```

### Change Hyperparameters
Edit the `models` dictionary in the respective module:

**For Classification (SVM kernel change):**
```python
'SVM': SVC(kernel='linear', ...)  # Instead of 'rbf'
```

**For Regression (more trees):**
```python
'Random Forest': RandomForestRegressor(n_estimators=200, ...)  # Instead of 100
```

---

## Troubleshooting

**Q: ImportError: No module named 'sklearn'**
```bash
pip install scikit-learn pandas numpy
```

**Q: Results different each time I run?**
A: Models still have randomness. Check `random_state` is set.

**Q: Accuracy/R² is low (< 0.5)?**
A: Model might need tuning. Try different hyperparameters or more data.

**Q: Script is slow?**
A: Reduce `n_estimators`, use `cv=3` instead of `cv=5`

---

## Next Steps

1. **Read DETAILED_EXPLANATION.md** for complete understanding
2. **Run examples** to see how they work
3. **Modify hyperparameters** and observe effects
4. **Try your own datasets** with these modules
5. **Combine with GridSearchCV** for automatic hyperparameter tuning
6. **Visualize results** with matplotlib for insights

---

For detailed explanations of every parameter, see **DETAILED_EXPLANATION.md**

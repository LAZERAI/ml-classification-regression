# ML Pipeline Modules - Complete Line-by-Line Explanation

## Table of Contents
1. [Classification Module](#classification-module)
2. [Regression Module](#regression-module)
3. [Key Concepts Explained](#key-concepts-explained)
4. [How to Use These Modules](#how-to-use-these-modules)
5. [Real-World Examples](#real-world-examples)

---

# CLASSIFICATION MODULE

## File: `classification_module_v2.py`

### IMPORTS SECTION

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
```

**Explanation:**
- `Pipeline`: A tool that chains multiple steps together (preprocessing + model training)
- `StandardScaler`: Normalizes features to have mean=0 and standard deviation=1
- `LogisticRegression`: A classification algorithm
- `DecisionTreeClassifier`: A tree-based classification algorithm
- `RandomForestClassifier`: An ensemble of decision trees
- `SVC`: Support Vector Classifier for classification
- `cross_val_score`: Evaluates model performance using cross-validation
- `accuracy_score`, `classification_report`, `confusion_matrix`: Metrics to evaluate models
- `numpy`: Math library for numerical operations

---

## CLASS: `ClassificationPipeline`

### `__init__` METHOD (Constructor)

```python
def __init__(self, model_name='Random Forest'):
```

**Line-by-line:**
- `def __init__(self, model_name='Random Forest'):` - Defines the constructor function
  - `self` - Reference to the object instance (required in Python)
  - `model_name='Random Forest'` - Parameter with default value 'Random Forest'
    - If user doesn't specify a model, it defaults to 'Random Forest'
    - Example: `pipeline = ClassificationPipeline()` uses 'Random Forest'
    - Example: `pipeline = ClassificationPipeline('SVM')` uses SVM

```python
    self.model_name = model_name
    self.pipeline = None
```

- `self.model_name = model_name` - Store the model name as an instance variable
  - `self.` means this belongs to the object (accessible later via `self.model_name`)
- `self.pipeline = None` - Initialize pipeline as empty (will be filled later)

### MODELS DICTIONARY

```python
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True
        )
    }
```

**This creates a dictionary with 4 different models. Let's break down each:**

#### Logistic Regression Model
```python
'Logistic Regression': LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

- `max_iter=1000` 
  - **What it is:** Maximum number of iterations for training
  - **What iterations are:** Training loops where the model adjusts its parameters
  - **Why 1000:** It's a safe number to ensure the model converges (learns properly)
  - **Analogy:** Like trying 1000 times to solve a puzzle until you get it right
  - **Why not more/less:** More takes longer, less might not be enough to learn properly

- `random_state=42`
  - **What it is:** Seed for randomness
  - **Why 42:** This specific number is arbitrary but fixed
  - **Why needed:** Makes results reproducible (you get the same answer each time you run)
  - **Without it:** Results would be slightly different on each run (due to randomness)
  - **Analogy:** Like planting a specific seed that always grows the same plant

#### Decision Tree Classifier
```python
'Decision Tree': DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

- `max_depth=3`
  - **What it is:** Maximum depth of the tree (number of levels)
  - **Why 3:** Limits complexity to prevent overfitting
  - **What overfitting is:** Model memorizes data instead of learning patterns
  - **How it works:** 
    - Depth 1: 1 split (1 decision)
    - Depth 2: Up to 2 splits (2 decisions)
    - Depth 3: Up to 3 splits (3 decisions)
  - **Why not deeper:** Deeper trees are more complex, memorize noise, perform worse on new data
  - **Visual example:**
    ```
    Depth 1:        [Root]
                    /    \
    Depth 2:     [A]      [B]
                 / \      / \
    Depth 3:   [C][D]   [E][F]
    ```

#### Random Forest Classifier
```python
'Random Forest': RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
```

- `n_estimators=100`
  - **What it is:** Number of decision trees in the forest
  - **Why 100:** A good balance between performance and speed
  - **How it works:** Creates 100 trees, each trained on random subsets of data
  - **Why multiple trees:** Reduces overfitting by averaging predictions
  - **Analogy:** Instead of asking 1 doctor, you ask 100 doctors and take majority vote
  - **Why not 1000:** More trees = slower training, minimal accuracy gain beyond ~100

- `max_depth=3` - Same as Decision Tree (prevents each individual tree from being too complex)

#### SVM (Support Vector Classifier)
```python
'SVM': SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42,
    probability=True
)
```

- **`kernel='rbf'`** - IMPORTANT! Let's explain kernels
  - **What it is:** How SVM learns decision boundaries
  - **Kernels available:**
    - `'linear'` - Draws straight lines to separate classes
      - Use when: Data is linearly separable
      - Example: Simple dataset where classes form linear patterns
    - `'rbf'` (Radial Basis Function) - Draws curved boundaries
      - Use when: Data has curved patterns (non-linear)
      - **Why RBF is chosen here:** Most flexible, works on many types of data
      - Works best when data forms clusters that aren't separable by lines
    - `'poly'` (Polynomial) - Draws polynomial curves
      - Use when: You want something between linear and RBF
    - `'sigmoid'` - Similar to neural network activation
      - Use when: Working with specific problem types
  - **Why RBF specifically:** It's the most robust choice that works for many problems
  - **Trade-off:** RBF is more flexible but slower than linear

- **`C=1.0`** - Regularization parameter
  - **What it is:** Controls how much error the model tolerates
  - **Higher C (e.g., 10.0):** Stricter, tries to classify every point correctly
    - Pros: Higher training accuracy
    - Cons: Overfitting risk
  - **Lower C (e.g., 0.1):** More lenient, allows some errors
    - Pros: Better generalization to new data
    - Cons: Underfitting risk
  - **Why 1.0:** Good default balance
  - **Why not 0 or negative:** 0 means no model learned, negatives don't make sense mathematically
  - **Common values:** 0.1, 1.0, 10, 100

- **`gamma='scale'`** - Controls influence of single training examples
  - **What it is:** How far each training point's influence extends
  - **Options:**
    - `'scale'` - Auto calculated (1 / (n_features * X.var())) → **RECOMMENDED**
    - `'auto'` - Old default (1 / n_features)
    - A number like 0.01, 0.1, etc.
  - **Higher gamma:** Each point has strong local influence
    - Effect: Complex, wiggly boundaries
    - Risk: Overfitting
  - **Lower gamma:** Each point influences farther distances
    - Effect: Smoother boundaries
    - Risk: Underfitting
  - **Why 'scale':** Automatically adapts to your data's scale

- **`probability=True`** - Enables probability estimates
  - **What it is:** Allows model to output confidence scores (0 to 1)
  - **Without it:** Model only outputs class label (e.g., "Iris Setosa")
  - **With it:** Model outputs probability (e.g., "95% sure it's Iris Setosa")
  - **Where used:** When you need confidence scores or probability estimates
  - **Trade-off:** Slightly slower training
  - **Why included:** Good practice for classification tasks

### PIPELINE CREATION

```python
    self.pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models[model_name])
    ])
```

**What it does:**
- Creates a two-step pipeline:
  1. **Step 1 - Scaler:** Normalizes features using StandardScaler
  2. **Step 2 - Model:** Applies the selected model
- **`Pipeline([...])` - List of tuples:**
  - First element: Name (string) - Used for reference
  - Second element: The actual transformer/estimator
- **Why this order:** Always scale BEFORE training model
  - Example: If features are [0-100, 0-10000], they're on different scales
  - StandardScaler makes them [mean=0, std=1] for fair comparison
  - Models like SVM are sensitive to scale; scaling helps them learn better

**Example visualization:**
```
Raw Data:          Scaled Data:        Model Prediction:
Feature1 Feature2  Feature1 Feature2   
100      10000     1.2      1.5        → Prediction: "Class A"
50       5000      -0.1     -0.1
75       7500      0.6      0.8
```

### PRINT STATEMENT

```python
    print(f"Classification pipeline created: {model_name}")
```

**Breaking down f-strings:**
- `print()` - Built-in function that outputs text to console
- `f"..."` - f-string (formatted string literal) - allows variable substitution
  - `f` prefix means "format this string"
- `{model_name}` - Placeholder for variable
  - At runtime, replaced with actual value
  - Example: If `model_name='SVM'`, outputs: "Classification pipeline created: SVM"
- **Why f-strings:**
  - **Readable:** Easy to see what's being inserted
  - **Fast:** More efficient than older methods
  - **Flexible:** Can include expressions: `f"Value: {x+5}"`
- **Alternative methods (older):**
  ```python
  print("Classification pipeline created: " + model_name)  # Concatenation
  print("Classification pipeline created: {}".format(model_name))  # .format()
  ```

---

## `train` METHOD

```python
def train(self, X_train, y_train, cv=5):
    print(f"\nTraining {self.model_name} model...")
    self.pipeline.fit(X_train, y_train)
```

**Parameters:**
- `self` - Object reference
- `X_train` - Training features (input data)
  - Shape example: (120, 4) = 120 samples, 4 features
- `y_train` - Training labels (target/answers)
  - Shape example: (120,) = 120 labels
- `cv=5` - Cross-validation folds (default 5)
  - **What it means:** Split data into 5 parts for validation
  - **Why 5:** Industry standard, good balance between accuracy and speed

**Line explanations:**
- `print(f"\nTraining {self.model_name} model...")`
  - `\n` - Newline character (moves to next line)
  - Why use it: Makes output more readable
  - Example output:
    ```
    (blank line)
    Training Random Forest model...
    ```

- `self.pipeline.fit(X_train, y_train)`
  - `fit()` - Train the pipeline on data
  - Process: Scaler learns normalization → Model learns patterns
  - What happens inside:
    1. StandardScaler calculates mean and std dev from X_train
    2. Scales X_train using these statistics
    3. Trains the model on scaled data
  - After this, model is ready to make predictions

```python
    cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring='accuracy')
```

- `cross_val_score()` - Evaluates model using k-fold cross-validation
- **How k-fold cross-validation works:**
  - Splits data into k parts (here k=5)
  - Trains k times: each time using k-1 parts for training, 1 part for testing
  - Returns k accuracy scores
  - **Why useful:** More reliable than single train-test split
  - **Example with cv=5:**
    ```
    Fold 1: Train on [2,3,4,5], Test on [1]
    Fold 2: Train on [1,3,4,5], Test on [2]
    Fold 3: Train on [1,2,4,5], Test on [3]
    Fold 4: Train on [1,2,3,5], Test on [4]
    Fold 5: Train on [1,2,3,4], Test on [5]
    
    Result: 5 accuracy scores (one per fold)
    ```
- `scoring='accuracy'` - Metric to use
  - For classification, 'accuracy' is standard
  - Other options: 'precision', 'recall', 'f1'

```python
    print(f"Training Complete")
    print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    return cv_scores.mean()
```

- `cv_scores.mean()` - Average of all 5 fold scores
- **`:.4f` - Format specification:**
  - `:` - Starts format specification
  - `4` - Show 4 decimal places
  - `f` - Fixed-point notation (float)
  - Example: `0.9567` instead of `0.956733819...`

- **`(+/- {cv_scores.std():.4f})` - Standard deviation:**
  - `std()` - Standard deviation (variation between folds)
  - Shows model consistency
  - Example output: `CV Score: 0.9567 (+/- 0.0234)`
    - Mean: 0.9567 (95.67% accuracy)
    - Std Dev: 0.0234 (variations up to ±2.34%)
    - Interpretation: Model performs consistently

---

## `evaluate` METHOD

```python
def evaluate(self, X_test, y_test, target_names=None):
    print(f"\nEvaluating {self.model_name} model...")
    y_pred = self.pipeline.predict(X_test)
```

**Parameter `target_names=None` - IMPORTANT:**
- **What it is:** List of class names for readability
- **Default (None):** If not provided, classes shown as numbers [0, 1, 2]
  - Example output:
    ```
              precision    recall  f1-score   support
               0       0.95      0.97      0.96        30
               1       0.92      0.91      0.91        32
               2       0.98      0.96      0.97        38
    ```
- **When provided:** Shows actual class names
  - Example: `target_names=['Setosa', 'Versicolor', 'Virginica']`
  - Output:
    ```
                  precision    recall  f1-score   support
            Setosa       0.95      0.97      0.96        30
        Versicolor       0.92      0.91      0.91        32
         Virginica       0.98      0.96      0.97        38
    ```
- **Why optional:** Works with any dataset, even if you don't know class names

```python
    accuracy = accuracy_score(y_test, y_pred)
```

- Calculates what % of predictions were correct
- Example: If 95/100 predictions correct, accuracy = 0.95

```python
    results = {
        'Accuracy': accuracy,
        'Predictions': y_pred,
        'Classification Report': classification_report(
            y_test, y_pred, target_names=target_names
        ),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
```

- Creates dictionary with 4 evaluation metrics:
  1. **Accuracy:** Overall correctness percentage
  2. **Predictions:** All predictions made
  3. **Classification Report:** Precision, recall, F1-score per class
  4. **Confusion Matrix:** How many correct/incorrect per class

```python
    print(f" Test Accuracy: {accuracy:.4f}")
    return results
```

---

## `predict` METHOD

```python
def predict(self, X):
    return self.pipeline.predict(X)
```

- **Simple wrapper method**
- Takes new data X and returns predictions
- Example: `predictions = pipeline.predict([[5.1, 3.5, 1.4, 0.2]])`

## `get_model_name` METHOD

```python
def get_model_name(self):
    return self.model_name
```

- Returns the model name stored during initialization
- Example: If created with `ClassificationPipeline('SVM')`, returns 'SVM'

---

---

# REGRESSION MODULE

## File: `regression_module_v2.py`

### IMPORTS SECTION

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
```

**Same as classification, but with regression models:**
- `LinearRegression`, `Ridge`, `Lasso` - Linear regression variants
- `RandomForestRegressor` - Ensemble for regression
- `SVR` - Support Vector Regressor
- `mean_squared_error`, `r2_score`, `mean_absolute_error` - Regression metrics

---

## CLASS: `RegressionPipeline`

### `__init__` METHOD

```python
def __init__(self, model_name='Random Forest'):
    self.model_name = model_name
    self.pipeline = None

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale')
    }
```

**Key differences from classification:**

#### Linear Regression
```python
'Linear': LinearRegression()
```
- **What it is:** Fits a straight line through data
- **Formula:** y = mx + b
- **No parameters:** Uses defaults
- **When to use:** Simplest baseline model
- **Trade-off:** Fast but limited (can only fit linear patterns)

#### Ridge Regression
```python
'Ridge': Ridge(alpha=1.0, random_state=42)
```

- **What it is:** Linear regression with regularization
- **`alpha=1.0` - Regularization strength:**
  - **What regularization does:** Prevents overfitting by penalizing large weights
  - **How:** Adds penalty to model when coefficients become too large
  - **Higher alpha (e.g., 10):** Stronger penalty, simpler model, less overfitting
  - **Lower alpha (e.g., 0.01):** Weaker penalty, more complex model, more overfitting
  - **Why 1.0:** Good balanced default
  - **Why we need it:** Without it, model might fit noise in training data
  - **Analogy:** Like forcing a student to study multiple topics equally (balanced) vs. letting them specialize (risky)

#### Lasso Regression
```python
'Lasso': Lasso(alpha=0.1, random_state=42)
```

- **Similar to Ridge but:**
  - Uses different penalty (L1 vs L2)
  - **Special property:** Can set some coefficients to exactly 0 (feature selection)
  - **Why alpha=0.1 (lower than Ridge):** Lasso is more aggressive; lower value to avoid over-simplifying
  - **Use case:** When you want to identify which features matter most

#### Random Forest Regressor
```python
'Random Forest': RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

- **Key difference:** `max_depth=10` instead of 3
- **Why deeper:** Regression needs to fit more complex continuous values
- **How it works:** Averages predictions from 100 trees instead of majority voting
- **Trade-off:** Deeper trees = better fit but slower, higher memory

#### SVR (Support Vector Regressor)
```python
'SVR': SVR(kernel='rbf', C=100, gamma='scale')
```

- **Similar to SVC but:**
  - For regression instead of classification
  - No `probability=True` (probabilities don't apply to regression)
  - `C=100` (higher than SVC's 1.0)
    - **Why higher:** SVR needs stronger regularization to fit curves properly
- **kernel='rbf':** Same as SVC - allows curved fitting

```python
    self.pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models[model_name])
    ])

    print(f"Regression pipeline created: {model_name}")
```

- **Same pipeline structure as classification**

---

## `train` METHOD

```python
def train(self, X_train, y_train, cv=5):
    print(f"\nTraining {self.model_name} model...")
    self.pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(
        self.pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='r2'
    )

    print("Training Complete")
    print(f"   CV R2 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return cv_scores.mean()
```

**Key difference:**
- `scoring='r2'` instead of 'accuracy'
  - **What R² is:** Coefficient of determination
  - **Range:** 0 to 1 (sometimes negative for bad models)
  - **Interpretation:**
    - R² = 0.95: Model explains 95% of variance
    - R² = 0.50: Model explains 50% of variance
    - R² = 0: Model is as bad as always predicting mean
  - **Why use it:** Standard metric for regression quality

---

## `evaluate` METHOD

```python
def evaluate(self, X_test, y_test):
    print(f"\nEvaluating {self.model_name} model...")
    y_pred = self.pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
```

**Explanation of regression metrics:**

- **MSE (Mean Squared Error):**
  - **Formula:** Average of (actual - predicted)²
  - **Range:** 0 to infinity
  - **Why square:** Penalizes large errors more heavily
  - **Example:** MSE=4 means average squared error is 4
  - **Lower is better**
  - **Disadvantage:** Hard to interpret (different units than data)

- **RMSE (Root Mean Squared Error):**
  - **Formula:** √MSE
  - **Range:** 0 to infinity
  - **Advantage:** Same units as target variable
  - **Example:** If predicting house prices in $, RMSE is also in $
  - **If RMSE=$50,000:** Average prediction off by $50,000
  - **Lower is better**

- **MAE (Mean Absolute Error):**
  - **Formula:** Average of |actual - predicted|
  - **Range:** 0 to infinity
  - **Advantage:** Easy to understand, doesn't over-penalize outliers
  - **Example:** MAE=$30,000 means average error is $30,000
  - **Lower is better**
  - **Difference from RMSE:** MAE treats all errors equally; RMSE penalizes large errors

- **R² (Coefficient of Determination):**
  - **Formula:** 1 - (Sum of squared errors) / (Total sum of squares)
  - **Range:** 0 to 1 (can be negative for terrible models)
  - **Interpretation:**
    - 1.0 = Perfect fit
    - 0.8-0.95 = Excellent
    - 0.5-0.8 = Good
    - 0.3-0.5 = Fair
    - <0.3 = Poor

**Metric comparison example:**
```
Actual:     [10, 20, 30]
Predicted:  [11, 19, 32]

MSE:   ((10-11)² + (20-19)² + (30-32)²) / 3 = (1 + 1 + 4) / 3 = 2.0
RMSE:  √2.0 = 1.41 (errors average 1.41 units)
MAE:   (|10-11| + |20-19| + |30-32|) / 3 = (1 + 1 + 2) / 3 = 1.33
R²:    0.98 (fits well)
```

```python
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2,
        'Predictions': y_pred
    }

    print(f" Test R2 Score: {r2:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE: {mae:.4f}")

    return results
```

- Returns all metrics for analysis

---

## `predict` and `get_model_name` METHODS

```python
def predict(self, X):
    return self.pipeline.predict(X)

def get_model_name(self):
    return self.model_name
```

- **Same as classification module**

---

---

# KEY CONCEPTS EXPLAINED

## Hyperparameters

**What are they?**
- Settings you choose BEFORE training (not learned from data)
- Control how the model learns

**Examples we've seen:**
- `max_iter=1000` - Number of training iterations
- `max_depth=3` - Tree depth
- `n_estimators=100` - Number of trees
- `C=1.0` - Regularization strength
- `alpha=1.0` - Regularization for Ridge/Lasso

**How to tune them:**
- Start with defaults (like we did)
- Change one at a time and observe effects
- Use GridSearchCV for systematic search

## Scaling/Normalization

**Why is StandardScaler important?**
- Puts all features on same scale
- Prevents features with larger ranges from dominating
- Required for algorithms like SVM, KNN, Neural Networks

**How it works:**
```
Original:   [1, 1000, 5]
            [2, 2000, 10]
            [3, 3000, 15]

Scaled:     [-1.22, -1.22, -1.22]
            [ 0,     0,     0   ]
            [ 1.22,  1.22,  1.22]

Formula: (x - mean) / std_dev
```

## Cross-Validation

**Why needed?**
- Single train-test split might be lucky
- Cross-validation uses multiple splits for reliable estimate
- More data used for training

**K-Fold process:**
```
Original Data: [1, 2, 3, 4, 5]

Fold 1: Train [2,3,4,5], Test [1]
Fold 2: Train [1,3,4,5], Test [2]
Fold 3: Train [1,2,4,5], Test [3]
Fold 4: Train [1,2,3,5], Test [4]
Fold 5: Train [1,2,3,4], Test [5]

Result: 5 accuracy scores → Average them
```

---

# HOW TO USE THESE MODULES

## Basic Usage Pattern

### For Classification:

```python
from classification_module_v2 import ClassificationPipeline

# Step 1: Create pipeline with chosen model
pipeline = ClassificationPipeline('Random Forest')

# Step 2: Train on training data
cv_score = pipeline.train(X_train, y_train, cv=5)

# Step 3: Evaluate on test data
results = pipeline.evaluate(X_test, y_test, target_names=['Class A', 'Class B'])

# Step 4: Make predictions on new data
predictions = pipeline.predict(new_data)

# Step 5: Get model name (optional)
print(pipeline.get_model_name())
```

### For Regression:

```python
from regression_module_v2 import RegressionPipeline

# Step 1: Create pipeline
pipeline = RegressionPipeline('Ridge')

# Step 2: Train
r2_score = pipeline.train(X_train, y_train)

# Step 3: Evaluate
results = pipeline.evaluate(X_test, y_test)

# Step 4: Predict
predictions = pipeline.predict(new_data)
```

---

# REAL-WORLD EXAMPLES

## Example 1: Iris Classification

See `example_iris_classification.py`

**What it does:**
1. Loads Iris dataset
2. Splits into train/test
3. Tests all 4 classification models
4. Compares results
5. Shows detailed metrics

**Expected output:**
- Training scores for each model
- Test accuracy for each model
- Confusion matrix
- Classification report with precision/recall

## Example 2: Iris Regression

See `example_iris_regression.py`

**What it does:**
1. Loads Iris dataset
2. Uses sepal length as target (to predict)
3. Splits into train/test
4. Tests all 5 regression models
5. Compares performance

**Expected output:**
- R² scores for each model
- RMSE values
- MAE values
- Actual vs predicted comparison

---

## SUMMARY TABLE: Key Parameters at a Glance

| Parameter | Purpose | Common Values | Range | Why Current Value |
|-----------|---------|----------------|-------|-------------------|
| `max_iter` | Training iterations | 100, 1000, 5000 | 1 to infinity | Ensures convergence |
| `max_depth` | Tree depth limit | 3-10 for trees | 1 to infinity | Prevents overfitting |
| `n_estimators` | Number of trees | 50-200 | 1 to infinity | Performance vs speed |
| `kernel` | SVM boundary type | 'linear', 'rbf', 'poly' | - | RBF most flexible |
| `C` | Regularization strength | 0.1, 1, 10, 100 | 0 to infinity | Higher = stricter |
| `gamma` | SVM influence range | 'scale', 'auto', 0.001-1 | 0 to infinity | 'scale' auto-adapts |
| `probability` | Enable probabilities | True, False | - | Needed for confidence |
| `alpha` | Regularization (Ridge/Lasso) | 0.01-10 | 0 to infinity | Controls overfitting |
| `random_state` | Reproducibility seed | 42, 0, 123 | - | Fixed = reproducible |
| `cv` | Cross-validation folds | 5, 10 | 2 to n_samples | Standard is 5 |
| `scoring` | Evaluation metric | 'accuracy', 'r2', 'f1' | - | Problem dependent |

---

## Common Troubleshooting

**Q: Why is accuracy/R² low?**
- A: Model might be too simple, try different model or more training data

**Q: Why do results change each run?**
- A: Missing `random_state` parameter; add it to make reproducible

**Q: Why is model slow?**
- A: Too many `n_estimators`, too large `max_depth`, or too large dataset

**Q: Why is model underfitting?**
- A: Model too simple; increase `max_depth`, `n_estimators`, or reduce `C`/`alpha`

**Q: Why is model overfitting?**
- A: Model too complex; decrease `max_depth`, increase `C`/`alpha`, or get more data

---

## Next Steps

1. **Experiment:** Try different models, change parameters
2. **Visualize:** Plot decision boundaries, confusion matrices
3. **Optimize:** Use GridSearchCV to find best parameters
4. **Deploy:** Save trained model, use in production
5. **Monitor:** Track performance over time

---

**Created:** January 2026 | **Version:** 2.0

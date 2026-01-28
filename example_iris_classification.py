"""
IRIS CLASSIFICATION EXAMPLE
============================
This script demonstrates how to use the ClassificationPipeline module
with the famous Iris dataset.

What it does:
1. Loads the Iris dataset (150 flowers, 4 measurements, 3 species)
2. Splits data into training (70%) and testing (30%)
3. Tests all 4 classification models
4. Compares their performance
5. Shows detailed evaluation metrics

Dataset Info:
- 150 samples (iris flowers)
- 4 features (sepal length, sepal width, petal length, petal width)
- 3 classes (Setosa, Versicolor, Virginica)
- Perfect for learning classification
"""

# ============================================================================
# IMPORTS
# ============================================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from classification_module_v2 import ClassificationPipeline
import pandas as pd
import numpy as np


def main():
    """Main function to run the classification example"""
    
    # ========================================================================
    # STEP 1: LOAD THE IRIS DATASET
    # ========================================================================
    
    print("=" * 70)
    print("IRIS CLASSIFICATION EXAMPLE")
    print("=" * 70)
    
    print("\n[STEP 1] Loading Iris dataset...")
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: 0=Setosa, 1=Versicolor, 2=Virginica
    target_names = iris.target_names  # Class names
    
    print(f"  ✓ Dataset loaded successfully!")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Feature names: {list(iris.feature_names)}")
    print(f"  - Number of classes: {len(target_names)}")
    print(f"  - Class names: {list(target_names)}")
    
    # Display first few samples
    print("\n  First 5 samples:")
    df_sample = pd.DataFrame(X[:5], columns=iris.feature_names)
    df_sample['Species'] = [target_names[i] for i in y[:5]]
    print(df_sample.to_string())
    
    # ========================================================================
    # STEP 2: SPLIT DATA INTO TRAINING AND TESTING
    # ========================================================================
    
    print("\n[STEP 2] Splitting data into train/test sets...")
    
    # test_size=0.3 means:
    #   - 70% (105 samples) for training
    #   - 30% (45 samples) for testing
    # random_state=42 ensures reproducible split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,      # Use 30% for testing
        random_state=42     # Fixed seed for reproducibility
    )
    
    print(f"  ✓ Data split successfully!")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples: {X_test.shape[0]}")
    print(f"  - Train/Test ratio: {X_train.shape[0]}/{X_test.shape[0]}")
    
    # ========================================================================
    # STEP 3: TEST ALL 4 CLASSIFICATION MODELS
    # ========================================================================
    
    print("\n[STEP 3] Training and evaluating all models...")
    print("-" * 70)
    
    # List of models to test
    models_to_test = [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'SVM'
    ]
    
    # Dictionary to store results for comparison
    all_results = {}
    
    # Train and evaluate each model
    for model_name in models_to_test:
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # CREATE PIPELINE
        pipeline = ClassificationPipeline(model_name)
        
        # TRAIN THE MODEL
        cv_score = pipeline.train(X_train, y_train, cv=5)
        
        # EVALUATE ON TEST DATA
        results = pipeline.evaluate(X_test, y_test, target_names=target_names)
        
        # STORE RESULTS
        all_results[model_name] = {
            'cv_score': cv_score,
            'test_accuracy': results['Accuracy'],
            'results': results
        }
        
        # DISPLAY CLASSIFICATION REPORT
        print("\n  Classification Report:")
        print(results['Classification Report'])
        
        # DISPLAY CONFUSION MATRIX
        print("\n  Confusion Matrix:")
        print("  (Rows = Actual, Columns = Predicted)")
        confusion = results['Confusion Matrix']
        print(f"            {target_names[0]:>12} {target_names[1]:>12} {target_names[2]:>12}")
        for i, class_name in enumerate(target_names):
            print(f"  {class_name:>12} {confusion[i][0]:>12} {confusion[i][1]:>12} {confusion[i][2]:>12}")
    
    # ========================================================================
    # STEP 4: COMPARE MODEL PERFORMANCE
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'CV Score': f"{results['cv_score']:.4f}",
            'Test Accuracy': f"{results['test_accuracy']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\n✓ Best performing model: {best_model[0]}")
    print(f"  Test Accuracy: {best_model[1]['test_accuracy']:.4f}")
    
    # ========================================================================
    # STEP 5: MAKE PREDICTIONS ON NEW DATA
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS ON NEW DATA")
    print("=" * 70)
    
    # Use the best model to make predictions
    best_pipeline = ClassificationPipeline(best_model[0])
    best_pipeline.train(X_train, y_train)
    
    # Example: Predict for first 5 test samples
    print(f"\nUsing {best_model[0]} to predict first 5 test samples:\n")
    
    predictions = best_pipeline.predict(X_test[:5])
    
    print("Sample | Feature Values                              | Actual       | Predicted")
    print("-" * 95)
    
    for i in range(5):
        features_str = f"[{X_test[i][0]:.1f}, {X_test[i][1]:.1f}, {X_test[i][2]:.1f}, {X_test[i][3]:.1f}]"
        actual = target_names[y_test[i]]
        predicted = target_names[predictions[i]]
        match = "✓" if actual == predicted else "✗"
        print(f"  {i+1}    | {features_str:<41} | {actual:<12} | {predicted:<12} {match}")
    
    # ========================================================================
    # STEP 6: EXPLAIN WHAT EACH METRIC MEANS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("UNDERSTANDING THE METRICS")
    print("=" * 70)
    
    print("""
CV Score (Cross-Validation Score):
  - Shows model performance during training
  - Split training data into 5 folds, tested 5 times
  - Average of 5 scores shown
  - Higher = Better (ranges 0 to 1)

Test Accuracy:
  - Shows model performance on unseen test data
  - Percentage of correct predictions
  - Higher = Better (ranges 0 to 1)
  - Example: 0.9333 = 93.33% correct predictions

Classification Report:
  - Precision: Of predicted positive cases, how many were correct?
  - Recall: Of actual positive cases, how many did we find?
  - F1-Score: Harmonic mean of precision and recall
  - Support: Number of samples in that class

Confusion Matrix:
  - Shows what the model predicted vs actual labels
  - Diagonal = Correct predictions
  - Off-diagonal = Mistakes
  - Example:
    [[15  0  0]
     [ 0 16  1]
     [ 0  0 14]]
    - 15 Setosa correctly predicted
    - 16 Versicolor correct, 1 wrongly predicted as Virginica
    - 14 Virginica correctly predicted
    """)
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
IRIS REGRESSION EXAMPLE
=======================
This script demonstrates how to use the RegressionPipeline module
with the Iris dataset used for regression.

What it does:
1. Loads the Iris dataset
2. Uses one flower measurement as the target to predict
3. Splits data into training (70%) and testing (30%)
4. Tests all 5 regression models
5. Compares their performance using regression metrics
6. Shows predictions vs actual values

Regression vs Classification:
- Classification: Predict category (Setosa, Versicolor, Virginica)
- Regression: Predict continuous value (exact measurement)

In this example:
- We predict "Petal Length" from other measurements
- Instead of "what species is this flower?"
- We ask "how long is this flower's petal?"
"""

# ============================================================================
# IMPORTS
# ============================================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from regression_module_v2 import RegressionPipeline
import pandas as pd
import numpy as np


def main():
    """Main function to run the regression example"""
    
    # ========================================================================
    # STEP 1: LOAD AND PREPARE IRIS DATASET FOR REGRESSION
    # ========================================================================
    
    print("=" * 80)
    print("IRIS REGRESSION EXAMPLE")
    print("=" * 80)
    
    print("\n[STEP 1] Loading Iris dataset and preparing for regression...")
    iris = load_iris()
    
    # Use all 4 features as input
    X = iris.data  # [sepal length, sepal width, petal length, petal width]
    
    # Predict: petal length (column 2)
    # We'll use features 0, 1, 3 (sepal length, sepal width, petal width)
    # and predict feature 2 (petal length)
    X_features = np.hstack([iris.data[:, [0, 1, 3]]])  # Remove petal length
    y_target = iris.data[:, 2]  # Petal length is our target
    
    print(f"  ✓ Dataset prepared for regression!")
    print(f"  - Number of samples: {X_features.shape[0]}")
    print(f"  - Input features: 3 (sepal length, sepal width, petal width)")
    print(f"  - Target variable: Petal Length (in cm)")
    print(f"  - Target range: {y_target.min():.1f} to {y_target.max():.1f} cm")
    print(f"  - Target mean: {y_target.mean():.2f} cm")
    
    # Display first few samples
    print("\n  First 5 samples:")
    df_sample = pd.DataFrame(
        X_features[:5],
        columns=['Sepal Length', 'Sepal Width', 'Petal Width']
    )
    df_sample['Petal Length (Target)'] = y_target[:5]
    print(df_sample.to_string())
    
    # ========================================================================
    # STEP 2: SPLIT DATA INTO TRAINING AND TESTING
    # ========================================================================
    
    print("\n[STEP 2] Splitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_target,
        test_size=0.3,      # Use 30% for testing (45 samples)
        random_state=42     # Fixed seed for reproducibility
    )
    
    print(f"  ✓ Data split successfully!")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples: {X_test.shape[0]}")
    print(f"  - Training target (petal length) range: {y_train.min():.1f} to {y_train.max():.1f} cm")
    print(f"  - Testing target (petal length) range: {y_test.min():.1f} to {y_test.max():.1f} cm")
    
    # ========================================================================
    # STEP 3: TEST ALL 5 REGRESSION MODELS
    # ========================================================================
    
    print("\n[STEP 3] Training and evaluating all regression models...")
    print("-" * 80)
    
    # List of models to test
    models_to_test = [
        'Linear',
        'Ridge',
        'Lasso',
        'Random Forest',
        'SVR'
    ]
    
    # Dictionary to store results for comparison
    all_results = {}
    
    # Train and evaluate each model
    for model_name in models_to_test:
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # CREATE PIPELINE
        pipeline = RegressionPipeline(model_name)
        
        # TRAIN THE MODEL
        r2_score = pipeline.train(X_train, y_train, cv=5)
        
        # EVALUATE ON TEST DATA
        results = pipeline.evaluate(X_test, y_test)
        
        # STORE RESULTS
        all_results[model_name] = {
            'r2_score': r2_score,
            'test_r2': results['R2 Score'],
            'rmse': results['RMSE'],
            'mae': results['MAE'],
            'mse': results['MSE'],
            'predictions': results['Predictions'],
            'results': results
        }
    
    # ========================================================================
    # STEP 4: COMPARE MODEL PERFORMANCE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - REGRESSION METRICS")
    print("=" * 80)
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'CV R²': f"{results['r2_score']:.4f}",
            'Test R²': f"{results['test_r2']:.4f}",
            'RMSE (cm)': f"{results['rmse']:.4f}",
            'MAE (cm)': f"{results['mae']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Find best model by R² score (higher is better)
    best_model = max(all_results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n✓ Best performing model: {best_model[0]}")
    print(f"  Test R² Score: {best_model[1]['test_r2']:.4f}")
    print(f"  RMSE: {best_model[1]['rmse']:.4f} cm")
    print(f"  MAE: {best_model[1]['mae']:.4f} cm")
    
    # ========================================================================
    # STEP 5: DETAILED ANALYSIS OF BEST MODEL
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: {best_model[0]}")
    print("=" * 80)
    
    best_pipeline = RegressionPipeline(best_model[0])
    best_pipeline.train(X_train, y_train)
    
    # Show first 10 predictions
    print("\nFirst 10 predictions vs actual values:")
    print("-" * 80)
    
    predictions = best_pipeline.predict(X_test[:10])
    
    print(f"{'Sample':<8} {'Sepal L':<10} {'Sepal W':<10} {'Petal W':<10} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 80)
    
    for i in range(10):
        actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        predicted = predictions[i]
        error = abs(actual - predicted)
        
        # Get features for display
        sepal_l = X_test[i, 0]
        sepal_w = X_test[i, 1]
        petal_w = X_test[i, 2]
        
        print(f"{i+1:<8} {sepal_l:<10.2f} {sepal_w:<10.2f} {petal_w:<10.2f} {actual:<10.2f} {predicted:<10.2f} {error:<10.2f}")
    
    # ========================================================================
    # STEP 6: CALCULATE AND SHOW ERROR STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    best_predictions = best_model[1]['predictions']
    errors = np.abs(y_test - best_predictions)
    
    print(f"\nError Statistics for {best_model[0]}:")
    print(f"  - Mean Error: {errors.mean():.4f} cm")
    print(f"  - Median Error: {np.median(errors):.4f} cm")
    print(f"  - Std Dev Error: {errors.std():.4f} cm")
    print(f"  - Min Error: {errors.min():.4f} cm")
    print(f"  - Max Error: {errors.max():.4f} cm")
    
    # Calculate percentage error
    percent_errors = (errors / y_test) * 100
    print(f"  - Mean % Error: {percent_errors.mean():.2f}%")
    
    # How many predictions are within certain thresholds?
    within_0_1 = (errors <= 0.1).sum()
    within_0_2 = (errors <= 0.2).sum()
    within_0_5 = (errors <= 0.5).sum()
    
    print(f"\nPredictions within threshold:")
    print(f"  - Within 0.1 cm: {within_0_1}/{len(errors)} ({within_0_1/len(errors)*100:.1f}%)")
    print(f"  - Within 0.2 cm: {within_0_2}/{len(errors)} ({within_0_2/len(errors)*100:.1f}%)")
    print(f"  - Within 0.5 cm: {within_0_5}/{len(errors)} ({within_0_5/len(errors)*100:.1f}%)")
    
    # ========================================================================
    # STEP 7: EXPLAIN REGRESSION METRICS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("UNDERSTANDING REGRESSION METRICS")
    print("=" * 80)
    
    print("""
R² Score (Coefficient of Determination):
  - Range: 0 to 1 (can be negative for very bad models)
  - Interpretation:
    * 1.0 = Perfect predictions
    * 0.9-1.0 = Excellent
    * 0.7-0.9 = Very Good
    * 0.5-0.7 = Good
    * 0.3-0.5 = Fair
    * <0.3 = Poor
  - Means: Explains what % of variation in target
  - Example: R²=0.85 means model explains 85% of the variance

RMSE (Root Mean Squared Error):
  - Units: Same as target (cm in this case)
  - Range: 0 to infinity (lower is better)
  - Calculation: √(average of squared errors)
  - Interpretation:
    * RMSE = 0.25 cm means average prediction error is 0.25 cm
    * Penalizes large errors more heavily
  - Use when: Large errors are especially bad
  - Example: If RMSE=0.25, expect ±0.25 cm error on average

MAE (Mean Absolute Error):
  - Units: Same as target (cm in this case)
  - Range: 0 to infinity (lower is better)
  - Calculation: average of |actual - predicted|
  - Interpretation:
    * MAE = 0.2 cm means average error is 0.2 cm
    * Treats all errors equally (doesn't penalize large errors extra)
  - Use when: All errors equally important
  - Difference from RMSE: More robust to outliers

MSE (Mean Squared Error):
  - Units: Squared target units (cm² in this case)
  - Range: 0 to infinity (lower is better)
  - Calculation: average of squared errors
  - Note: Usually converted to RMSE for interpretability
  - Relationship: RMSE = √MSE

Comparing Models:
  - Higher R² = Better
  - Lower RMSE = Better
  - Lower MAE = Better
  - All three should generally agree on model ranking
    """)
    
    # ========================================================================
    # STEP 8: SHOW HOW HYPERPARAMETERS AFFECT DIFFERENT MODELS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("HOW DIFFERENT MODELS WORK")
    print("=" * 80)
    
    print("""
Linear Regression:
  - Fits a straight line through data
  - Fast, simple, interpretable
  - Limited: can only capture linear relationships
  - No hyperparameters to tune
  - Best for: Simple linear problems

Ridge Regression (alpha=1.0):
  - Linear regression with penalty on large coefficients
  - Prevents overfitting
  - alpha = regularization strength (higher = simpler model)
  - Best for: Linear patterns with many correlated features

Lasso Regression (alpha=0.1):
  - Linear regression that can set coefficients to zero
  - Does feature selection (identifies important features)
  - alpha lower than Ridge because it's more aggressive
  - Best for: Feature selection, sparse solutions

Random Forest (n_estimators=100, max_depth=10):
  - Ensemble of 100 decision trees
  - Each tree predicts, then averages all predictions
  - max_depth=10 allows deeper trees (more complex patterns)
  - Robust to outliers
  - Best for: Complex non-linear patterns

SVR (Radial Basis Function kernel, C=100):
  - Support Vector Regressor with RBF kernel
  - Creates curved decision boundaries
  - C=100 is high (strict, less tolerance for errors)
  - Gamma controls influence range
  - Best for: Complex non-linear patterns
    """)
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    # Return results for further analysis if needed
    return all_results, X_test, y_test


if __name__ == "__main__":
    results, X_test, y_test = main()

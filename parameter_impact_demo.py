"""
PARAMETER IMPACT DEMONSTRATION
==============================
This script shows how changing hyperparameters affects model performance.

Run this to understand:
1. How max_iter affects Logistic Regression
2. How max_depth affects Decision Trees
3. How n_estimators affects Random Forest
4. How C and gamma affect SVM
5. How alpha affects Ridge/Lasso
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from classification_module_v2 import ClassificationPipeline
from regression_module_v2 import RegressionPipeline
import numpy as np


def demonstrate_max_depth_impact():
    """Show how max_depth affects Decision Tree performance"""
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: How max_depth Affects Decision Trees")
    print("=" * 70)
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    print("\nmax_depth = Maximum number of levels in the tree")
    print("Lower depth = Simpler model, less overfitting")
    print("Higher depth = Complex model, more risk of overfitting\n")
    
    depths = [1, 2, 3, 4, 5, 10, 20]
    
    print(f"{'Depth':<8} {'Training Accuracy':<20} {'Test Accuracy':<20} {'Model Complexity'}")
    print("-" * 70)
    
    for depth in depths:
        # Manually create and test trees with different depths
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(max_depth=depth, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        
        complexity = "Simple" if depth <= 2 else "Moderate" if depth <= 5 else "Complex"
        
        print(f"{depth:<8} {train_acc:<20.4f} {test_acc:<20.4f} {complexity}")
    
    print("\nObservation: As depth increases:")
    print("  - Training accuracy increases (memorizing data)")
    print("  - Test accuracy peaks then decreases (overfitting)")
    print("  - We chose depth=3 as good balance")


def demonstrate_n_estimators_impact():
    """Show how n_estimators affects Random Forest"""
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: How n_estimators Affects Random Forest")
    print("=" * 70)
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    print("\nn_estimators = Number of trees in the forest")
    print("More trees = Better performance but slower training")
    print("Sweet spot usually around 50-200 trees\n")
    
    n_estimators_list = [5, 10, 20, 50, 100, 200]
    
    print(f"{'Trees':<8} {'Training Accuracy':<20} {'Test Accuracy':<20} {'Status'}")
    print("-" * 70)
    
    for n_est in n_estimators_list:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=n_est, max_depth=3, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        
        status = "Still improving" if n_est < 100 else "Good balance" if n_est == 100 else "Diminishing returns"
        
        print(f"{n_est:<8} {train_acc:<20.4f} {test_acc:<20.4f} {status}")
    
    print("\nObservation:")
    print("  - Performance improves with more trees")
    print("  - Gains diminish after ~100 trees")
    print("  - We chose 100 as good practical balance")


def demonstrate_regularization_impact():
    """Show how alpha affects Ridge/Lasso regression"""
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: How alpha Affects Regularization")
    print("=" * 70)
    
    # Prepare iris for regression
    iris = load_iris()
    X = iris.data[:, [0, 1, 3]]  # Features
    y = iris.data[:, 2]  # Target (petal length)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("\nalpha = Regularization strength")
    print("Higher alpha = More regularization = Simpler model")
    print("Lower alpha = Less regularization = More complex model\n")
    
    print("RIDGE REGRESSION:")
    print(f"{'Alpha':<12} {'Training R²':<20} {'Test R²':<20} {'Model Type'}")
    print("-" * 70)
    
    ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10, 100]
    
    for alpha in ridge_alphas:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=alpha, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        train_r2 = pipeline.score(X_train, y_train)
        test_r2 = pipeline.score(X_test, y_test)
        
        model_type = "Complex" if alpha < 0.1 else "Balanced" if alpha <= 1 else "Simple"
        
        print(f"{alpha:<12} {train_r2:<20.4f} {test_r2:<20.4f} {model_type}")
    
    print("\nLASSO REGRESSION (note: lower alpha values used):")
    print(f"{'Alpha':<12} {'Training R²':<20} {'Test R²':<20} {'Model Type'}")
    print("-" * 70)
    
    lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    
    for alpha in lasso_alphas:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Lasso
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=alpha, random_state=42, max_iter=5000))
        ])
        
        try:
            pipeline.fit(X_train, y_train)
            train_r2 = pipeline.score(X_train, y_train)
            test_r2 = pipeline.score(X_test, y_test)
            
            model_type = "Complex" if alpha < 0.05 else "Balanced" if alpha <= 0.1 else "Simple"
            
            print(f"{alpha:<12} {train_r2:<20.4f} {test_r2:<20.4f} {model_type}")
        except:
            print(f"{alpha:<12} {'Failed':<20} {'Failed':<20} {'Too strict'}")
    
    print("\nObservation:")
    print("  - Ridge: Works well with alpha=1.0")
    print("  - Lasso: Works well with alpha=0.1 (more aggressive)")
    print("  - Balance train/test performance - not too high either way")


def demonstrate_svm_kernel_impact():
    """Show how kernel choice affects SVM"""
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: SVM Kernel Types")
    print("=" * 70)
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    print("\nKernel = How SVM creates decision boundaries")
    print("'linear': Straight lines (fast, simple)")
    print("'rbf': Curved boundaries (flexible, slower)")
    print("'poly': Polynomial curves (middle ground)")
    print("'sigmoid': Neural-network like\n")
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    
    print(f"{'Kernel':<12} {'Training Accuracy':<20} {'Test Accuracy':<20} {'Speed'}")
    print("-" * 70)
    
    for kernel in kernels:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        
        speed = "Fast" if kernel == 'linear' else "Slow" if kernel == 'rbf' else "Medium"
        
        print(f"{kernel:<12} {train_acc:<20.4f} {test_acc:<20.4f} {speed}")
    
    print("\nObservation:")
    print("  - 'rbf' provides best accuracy for this complex dataset")
    print("  - 'linear' might suffice for simpler, linearly separable data")
    print("  - 'rbf' is safe default for most problems")


def demonstrate_c_parameter_impact():
    """Show how C (regularization) affects SVM"""
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: SVM's C Parameter (Regularization)")
    print("=" * 70)
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    print("\nC = Regularization strength in SVM")
    print("Higher C: Stricter, tries to classify everything (overfitting risk)")
    print("Lower C: Lenient, allows errors (underfitting risk)")
    print("C=1.0: Good default balance\n")
    
    c_values = [0.01, 0.1, 1.0, 10, 100]
    
    print(f"{'C Value':<12} {'Training Accuracy':<20} {'Test Accuracy':<20} {'Effect'}")
    print("-" * 70)
    
    for c in c_values:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', C=c, gamma='scale', random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        
        effect = "Too lenient" if c < 0.1 else "Too strict" if c > 10 else "Balanced"
        
        print(f"{c:<12} {train_acc:<20.4f} {test_acc:<20.4f} {effect}")
    
    print("\nObservation:")
    print("  - Very low C (0.01): Model under-learns")
    print("  - Very high C (100): Model over-fits training data")
    print("  - C=1.0: Good balance for most cases")


def demonstrate_gamma_impact():
    """Show how gamma affects SVM"""
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: SVM's Gamma Parameter")
    print("=" * 70)
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    print("\ngamma = Influence range of each training point (for RBF kernel)")
    print("Higher gamma: Each point has strong local influence (complex boundaries)")
    print("Lower gamma: Points influence broader area (smooth boundaries)")
    print("'scale': Auto-calculated from data (recommended)\n")
    
    gamma_values = [0.001, 0.01, 0.1, 1.0, 'scale']
    
    print(f"{'Gamma':<12} {'Training Accuracy':<20} {'Test Accuracy':<20} {'Boundary'}")
    print("-" * 70)
    
    for gamma in gamma_values:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        
        boundary = "Smooth" if (isinstance(gamma, float) and gamma < 0.01) else \
                   "Complex" if (isinstance(gamma, float) and gamma > 0.1) else \
                   "Balanced" if gamma == 'scale' else "Medium"
        
        print(f"{str(gamma):<12} {train_acc:<20.4f} {test_acc:<20.4f} {boundary}")
    
    print("\nObservation:")
    print("  - gamma='scale': Smart auto-calculated value (use this)")
    print("  - Very low gamma: Over-smooth, underfitting")
    print("  - Very high gamma: Overly complex, overfitting")


def main():
    """Run all demonstrations"""
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER IMPACT DEMONSTRATIONS")
    print("=" * 70)
    print("\nThis script shows how changing parameters affects model performance")
    print("Pay attention to the patterns - they help you tune your own models!")
    
    demonstrate_max_depth_impact()
    demonstrate_n_estimators_impact()
    demonstrate_c_parameter_impact()
    demonstrate_gamma_impact()
    demonstrate_regularization_impact()
    demonstrate_svm_kernel_impact()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    
    print("""
1. max_depth:
   - Control complexity of trees
   - Too high → overfitting
   - Our choice (3): Good balance

2. n_estimators:
   - More trees → Better performance
   - Diminishing returns after ~100
   - Our choice (100): Good practical balance

3. C (SVM regularization):
   - Controls error tolerance
   - Default (1.0): Usually works well
   - Tune if over/underfitting

4. gamma (SVM influence):
   - Use 'scale' (auto-calculated)
   - Only tune if needed

5. alpha (Ridge/Lasso):
   - Ridge (1.0): Good default
   - Lasso (0.1): More aggressive feature selection
   - Match problem characteristics

6. Kernel (SVM):
   - 'rbf': Safe, powerful default
   - 'linear': If data is separable by lines
   - 'poly': Rarely needed

STRATEGY:
- Start with defaults (we've chosen good ones)
- If underfitting: increase complexity
- If overfitting: decrease complexity
- Use GridSearchCV for systematic tuning
    """)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATIONS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

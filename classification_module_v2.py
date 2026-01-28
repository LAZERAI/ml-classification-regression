from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


class ClassificationPipeline:
    def __init__(self, model_name='Random Forest'):
        
        self.model_name = model_name
        self.pipeline = None
    
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
                n_estimators=50,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=3,
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
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[model_name])
        ])
        
        print(f"Classification pipeline created: {model_name}")

    def train(self, X_train, y_train, cv=5):
        print(f"\nTraining {self.model_name} model...")
        self.pipeline.fit(X_train, y_train)

        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, scoring='accuracy')

        print(f"Training Complete")
        print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return cv_scores.mean()
    
    def evaluate(self, X_test, y_test, target_names=None):
        print(f"\nEvaluating {self.model_name} model...")
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results = {
            'Accuracy': accuracy,
            'Predictions': y_pred,
            'Classification Report': classification_report(
                y_test, y_pred, target_names=target_names
            ),
            'Confusion Matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f" Test Accuracy: {accuracy:.4f}")
        
        return results
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def get_model_name(self):
        return self.model_name

    def plot_model_accuracies(self, results_dict):
        model_names = list(results_dict.keys())
        accuracies = list(results_dict.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(model_names, accuracies)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)

        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.show()

    def test_all_models(self, X_train, X_test, y_train, y_test, target_names=None):
        all_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
        results = {}

        for model_name in all_models:
            print(f"\nTesting {model_name}...")
            
            model = ClassificationPipeline(model_name)
            model.train(X_train, y_train)
            result = model.evaluate(X_test, y_test, target_names=target_names)
            
            results[model_name] = result['Accuracy']
            print(f"Confusion Matrix:")
            print(result['Confusion Matrix'])
            print(f"\nClassification Report:")
            print(result['Classification Report'])
        
        print("\n" + "="*50)
        print("All Results:")
        print("="*50)
        for model_name in results:
            print(f"{model_name}: {results[model_name]:.4f}")
        
        return results
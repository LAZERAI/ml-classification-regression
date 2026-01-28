from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


class RegressionPipeline:
    def __init__(self, model_name='Random Forest'):
        
        self.model_name = model_name
        self.pipeline = None

        models = {
            'Linear': LinearRegression(),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=100, gamma='scale')
        }

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[model_name])
        ])

        print(f"Regression pipeline created: {model_name}")

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

    def evaluate(self, X_test, y_test):
        print(f"\nEvaluating {self.model_name} model...")
        y_pred = self.pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

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

    def predict(self, X):
        return self.pipeline.predict(X)
    
    def get_model_name(self):
        return self.model_name

    def plot_model_r2(self, results_dict):
        model_names = list(results_dict.keys())
        r2_scores = [results_dict[name]['r2'] for name in model_names]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(model_names, r2_scores)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)

        plt.xlabel("Models")
        plt.ylabel("R² Score")
        plt.title("Model R² Score Comparison")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.show()

    def test_all_models(self, X_train, X_test, y_train, y_test):
        all_models = ['Linear', 'Gradient Boosting', 'Random Forest', 'SVR']
        results = {}

        for model_name in all_models:
            print(f"\nTesting {model_name}...")
            
            model = RegressionPipeline(model_name=model_name)
            model.train(X_train, y_train)
            result = model.evaluate(X_test, y_test)
            
            results[model_name] = {
                'r2': result['R2 Score'],
                'rmse': result['RMSE'],
                'mae': result['MAE']
            }
        
        print("\n" + "="*50)
        print("All Results:")
        print("="*50)
        for model_name in results:
            r2 = results[model_name]['r2']
            rmse = results[model_name]['rmse']
            mae = results[model_name]['mae']
            print(f"{model_name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        return results
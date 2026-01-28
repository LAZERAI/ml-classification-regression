import pandas as pd
from sklearn.model_selection import train_test_split
from regression_module_v2 import RegressionPipeline

df = pd.read_csv(r"C:\Users\Lazerai\Downloads\Dataset-20260127T101434Z-3-001\Dataset\car_data.csv")

X = df[['Engine_Size(L)', 'Horsepower', 'Weight(kg)', 'MPG']].values
y = df['Price($1000)'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RegressionPipeline()
results_all = model.test_all_models(X_train, X_test, y_train, y_test)

model.plot_model_r2(results_all)
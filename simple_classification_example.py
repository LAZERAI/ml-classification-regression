import pandas as pd
from sklearn.model_selection import train_test_split
from classification_module_v2 import ClassificationPipeline

df = pd.read_csv(r"C:\Users\Lazerai\Downloads\Dataset-20260127T101434Z-3-001\Dataset\diabetes2.csv")

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = ClassificationPipeline()
results_all = model.test_all_models(
    X_train, X_test, y_train, y_test, 
    target_names=['No Diabetes', 'Diabetes']
)

model.plot_model_accuracies(results_all)
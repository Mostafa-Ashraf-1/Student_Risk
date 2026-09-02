
import os
import subprocess
import sys

# Force pip to install mlflow and dagshub dynamically at runtime
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mlflow", "dagshub"])


# Now you can safely import it
import mlflow
import dagshub
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


os.environ["MLFLOW_TRACKING_USERNAME"] = "Mostafa-Ashraf-1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
mlflow.set_tracking_uri('https://dagshub.com/Mostafa-Ashraf-1/Student_Risk.mlflow')


train_df = pd.read_csv('/kaggle/input/playground-series-s6e7/train.csv')

train_df.drop(columns = ['id'], inplace=True)

X = train_df.drop(columns=['health_condition'])
y = train_df['health_condition']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y, random_state=42)

num_cols = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake"
]

cat_cols = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender"
]

# Numeric preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

"""
For feature engineering > We need to define a class that handles the columns
and pass that class into our pipeline
"""

MODEL_REGISTERY ={
     'LogisticRegression' : LogisticRegression,
     'RandomForest' : RandomForestClassifier
}

with open('/kaggle/input/StudentRisk_PARAMS/params.yaml', 'r') as f:
    models = yaml.safe_load(f)['train']['model']

classifier = 'RandomForest'
params = models[classifier]

#mlflow.set_tracking_uri('https://dagshub.com/Mostafa-Ashraf-1/Student_Risk.mlflow')
mlflow.set_experiment("Student-Risk-Model")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name=f'Student-Risk/{classifier}'):
        
        pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", MODEL_REGISTERY[classifier]( **params ))
            ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_valid)

        val_f1 = f1_score(y_valid, y_pred, average='weighted')
        val_accuracy = accuracy_score(y_valid, y_pred)

        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("val_accuracy", val_accuracy)



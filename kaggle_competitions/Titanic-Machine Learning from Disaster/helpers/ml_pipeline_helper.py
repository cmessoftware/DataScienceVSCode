import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

class MachineLearningPipelineHelper:
    def __init__(self, features, model=None):
        self.features = features
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocess(self, df):
        df = df.copy()
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        return df

    def train(self, X, y):
        self.model.fit(X[self.features], y)

    def predict(self, X):
        return self.model.predict(X[self.features])

    def evaluate_local(self, X, y, test_size=0.2):
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X[self.features], y, test_size=test_size, random_state=42
        )
        self.model.fit(X_train_split, y_train_split)
        y_pred_val = self.model.predict(X_val_split)
        acc = accuracy_score(y_val_split, y_pred_val)
        print(f"🔎 Accuracy local en validación: {acc:.4f}")
        return acc

    def save_submission(self, passenger_ids, predictions, filename='submission.csv'):
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        submission.to_csv(filename, index=False)

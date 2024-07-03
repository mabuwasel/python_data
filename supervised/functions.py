import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Models to compare
models = {
'Decision Tree': DecisionTreeClassifier(),
'Random Forest': RandomForestClassifier(),
'Gradient Boosting': GradientBoostingClassifier(),
'SVM': SVC(),
}


def models_cv(models: dict,X: pd.DataFrame,y: pd.Series,test_size: float = 0.2,
              random_state: int = 42,cat_features: list = None):

    if cat_features:
        label_encoders = {}
        for column in cat_features:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            # saving encoders for later decoding
            label_encoders[column] = le
        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate models
    for name, model in models.items():
        if name in ['SVM', 'Neural Network']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        print(f'{name} Model:')
        print(classification_report(y_test, y_pred))

    # Cross-validation for model selection - default 5 folds
    for name, model in models.items():
        if name in ['SVM', 'Neural Network']:
            scores = cross_val_score(model, X_train_scaled, y_train)
        else:
            scores = cross_val_score(model, X_train, y_train)
        
        print(f'{name} Model Cross-Validation Accuracy: {scores.mean()}')
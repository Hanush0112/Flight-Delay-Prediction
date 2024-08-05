import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Data/Processed_data15.csv')

# Label Encoding
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Prepare features and target variable
X = df.iloc[:, 0:6].values  # Adjust if needed based on your feature columns
y = df['delayed']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=61)

# Initialize RandomForestClassifier
model = RandomForestClassifier(random_state=61)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Save the best model
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Save label encoders
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump({
        'carrier': le_carrier,
        'dest': le_dest,
        'origin': le_origin
    }, le_file)

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with best parameters: {accuracy:.2f}")

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Feature Importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature Importances:")
for i in range(X.shape[1]):
    print(f"Feature {i + 1}: {indices[i]} (importance: {importances[indices[i]]:.4f})")

print("Best model parameters:", grid_search.best_params_)

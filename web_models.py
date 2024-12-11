import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the dataset
df1 = pd.read_csv('diabetes.csv')

# Drop specific columns
columns_to_remove = [2, 3, 4, 6]  # Example indices of columns to remove
df_dropped = df1.iloc[:, ~df1.columns.isin(df1.columns[columns_to_remove])]

# Split into features and target
A = df_dropped.iloc[:, :-1]
b = df_dropped.iloc[:, -1]

# Train-test split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=2)

# Apply SMOTE for handling imbalanced classes
smote = SMOTE(random_state=2)
A_train_resampled, b_train_resampled = smote.fit_resample(A_train, b_train)

# Standardize features
scaler = StandardScaler()
A_train_scaled = scaler.fit_transform(A_train_resampled[['Pregnancies', 'Glucose', 'BMI', 'Age']])
A_test_scaled = scaler.transform(A_test[['Pregnancies', 'Glucose', 'BMI', 'Age']])
joblib.dump(scaler, 'scaler.pkl')

# Train a RandomForestClassifier
rfc = RandomForestClassifier(max_depth=None, 
                             min_samples_leaf=1, 
                             min_samples_split=5, 
                             n_estimators=100)
rfc.fit(A_train_scaled, b_train_resampled)

# Make predictions
nb_pred = rfc.predict(A_test_scaled)

# Print classification report and confusion matrix
print(classification_report(b_test, nb_pred))
print(confusion_matrix(b_test, nb_pred))

# Save the trained model
with open('trained_model.sav', 'wb') as model_file:
    pickle.dump(rfc, model_file)

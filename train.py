import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


sample_df = pd.read_pickle("sample_df.pkl")
print('sample dataframe loaded!!')

sample_df.drop(columns=['nameOrig', 'nameDest', 'step'], inplace=True)

# Define the feature columns and target
features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
target = 'isFraud'

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['type']),  # One-hot encode 'A' and 'B'
        ('scaler', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])       # Standard scale 'C'
    ],
    remainder='passthrough'  # Keep the rest of the columns unchanged
)
print('transformation class created!!')

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
print('model pipeline is created!!')

# Separate features and target
X = sample_df[features]
y = sample_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=21)

# Fit the pipeline to the data
pipeline.fit(X_train, y_train)
print('model training completed!!')

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Load the model from the pickle file
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)
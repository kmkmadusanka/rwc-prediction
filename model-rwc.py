import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the noisy dataset
data = pd.read_csv('noisy_autos_0_1.csv')

# Preprocess the dataset: Encode categorical variables
data_encoded = pd.get_dummies(data[['yearOfRegistration', 'kilometer', 'brake_condition', 'tire_condition',
                                    'suspension_condition', 'emission_compliance']], drop_first=True)

# Print the column names to check what has been encoded
print("Encoded columns:", data_encoded.columns.tolist())

# After seeing the column names, adjust the target column
target_column = [col for col in data_encoded.columns if 'brake_condition' in col][0]
print(f"Using target column: {target_column}")

# Define X (features) and y (target)
X = data_encoded.drop(columns=[target_column])  # Dropping the target column from features
y = data_encoded[target_column]  # Target is the dynamically selected brake_condition column

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 1: Instantiate the RandomForest model
clf = RandomForestClassifier(random_state=42)

# Step 2: Perform 5-fold cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores and the mean score
print(f"Cross-validation scores for each fold: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

# Step 3: Train the model on the entire training set
clf.fit(X_train, y_train)

# Step 4: Evaluate the model on the test set
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 5: Save the trained model to a pickle file
with open('roadworthy_model_with_noise.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

print("Model training complete and saved.")

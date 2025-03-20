import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import joblib

# Load datasets
responses_file = "Responses.csv"
noisy_dataset_file = "Noisy_Extended_Dataset1.csv"

responses_df = pd.read_csv(responses_file)
noisy_df = pd.read_csv(noisy_dataset_file)

# Standardizing column names (removing extra spaces and making them consistent)
responses_df.columns = responses_df.columns.str.strip()
noisy_df.columns = noisy_df.columns.str.strip()

# Fix column name mismatches
noisy_df.rename(columns={'Sleep dissorder': 'Sleep Dissorder', 
                         'Suicidal thoughts': 'Suicidal Thought'}, inplace=True)

# Convert categorical values like "3 From 10" to numerical
def convert_to_numeric(value):
    if isinstance(value, str) and "From" in value:
        return int(value.split()[0])  # Extracts the first number
    return value

numeric_columns = ["Sexual Activity", "Concentration", "Optimisim"]
for col in numeric_columns:
    noisy_df[col] = noisy_df[col].apply(convert_to_numeric)

# Encode binary "Yes"/"No" responses
binary_columns = ["Mood Swing", "Suicidal Thought", "Anorxia", "Authority Respect", 
                  "Try-Explanation", "Aggressive Response", "Ignore & Move-On", 
                  "Nervous Break-down", "Admit Mistakes", "Overthinking"]

for col in binary_columns:
    noisy_df[col] = noisy_df[col].map({"YES": 1, "NO": 0, "Yes": 1, "No": 0})

# Encode ordinal responses
ordinal_columns = ["Sadness", "Euphoric", "Exhausted", "Sleep Dissorder"]
ordinal_mapping = {"Seldom": 1, "Sometimes": 2, "Usually": 3, "Most Often": 4, "Most-Often": 4}

for col in ordinal_columns:
    noisy_df[col] = noisy_df[col].map(ordinal_mapping)

# Encode target variable "Expert Diagnose"
label_encoder = LabelEncoder()
noisy_df["Expert Diagnose"] = label_encoder.fit_transform(noisy_df["Expert Diagnose"])

# Handle missing values
for col in numeric_columns:
    noisy_df[col].fillna(noisy_df[col].median(), inplace=True)

for col in binary_columns + ordinal_columns:
    noisy_df[col].fillna(noisy_df[col].mode()[0], inplace=True)

# Train-test split
X = noisy_df.drop(columns=["Expert Diagnose"])
y = noisy_df["Expert Diagnose"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train, y_train)

# Predict on test set
y_pred = knn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
report=classification_report(y_test,y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)


#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Save trained model
joblib.dump(knn_model, "mental_health_knn_model.pkl")

# Save label encoder and scaler separately
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")


print("numpy",np.__version__)
print("Pandas",pd.__version__)
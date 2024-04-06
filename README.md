# Heart-disease-predication- git clone https://github.com/your-username/heart-disease-prediction.git
import pandas as pd

# Load the dataset from CSV file
df = pd.read_csv("heart_disease_dataset.csv")

# Display the first few rows of the dataset
print(df.head())

# Explore basic statistics of the dataset
print(df.describe())

# Separate features (X) and target variable (y)
X = df.drop(columns=['target'])  # Features (attributes)
y = df['target']  # Target variable (presence or absence of heart disease)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (e.g., Decision Tree Classifier)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
git add .
git commit -m "Initial commit: Add Python code files"
git push origin master

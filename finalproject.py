# 1. Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE



# 2. Load Data

# Read the dataset file and store it in a DataFrame
data = pd.read_csv('templates/oasis_train_patients_metadata.csv')


# 3. Drop Unnecessary Columns : that do not help the project

# keep the needed data and remove the rest
colums_to_drop=['ID','Hand','Educ','SES','eTIV','nWBV','ASF','Delay']
data=data.drop(columns=colums_to_drop)





# 5. Basic Info

print(data.head()) # First 5 rows
print(data.info()) # Data types
print(data.isnull().sum()) # Missing values in each column
print(data["class"].value_counts()) # Distribution of class labels (NonDemented, Very Mild Demented, Moderate Demented, or Mild Demented )
print(data[['Age', 'MMSE','CDR']].describe()) # Summary statistics for numerical features --> max , min , std



# 6. Visualizations (Graphs and Charts)

main_color = "#6A5ACD"

# Class Distribution
#Seaborn library / countplot() creates a bar chart
sns.countplot(x='class', data=data, color=main_color)
plt.title("Class Distribution")   # Graph title
plt.show()                        # Show graph


# MMSE Distribution
#Seaborn library / Creates a histogram for MMSE values
sns.histplot(data['MMSE'], kde=True, color=main_color)# kde=True : adds a smooth curve line
plt.title("MMSE Distribution")    # Graph title
plt.show()                        # Show graph


# Missing Values
mv = data.isnull().sum()          # Count missing values
#Seaborn library / barplot() creates a bar chart
sns.barplot(x=mv.index, y=mv.values, color=main_color)
plt.title("Missing Values Count") # Graph title
plt.ylabel("Number of Missing Values")  # Y-axis label
plt.show()                        # Show graph




# 7. Handling Missing Values

# Missing values in MMSE and CDR are filled using the mean value.
# Mean is usually used when the data is normally distributed
# and there are no extreme outliers.

# Median can also be used if the data contains outliers
# because it is less affected by very high or very low values.

data["MMSE"] = data["MMSE"].fillna(data["MMSE"].mean())
data["CDR"] = data["CDR"].fillna(data["CDR"].mean())




# 8. Encoding Gender

# Gender may not be a very important feature for prediction,
# but it can help in statistical analysis,
# such as comparing the number of patients
# between males and females.

# Machine learning models cannot work with text values directly,
# so gender values are converted into numbers.

# M (Male)  -> 0
# F (Female) -> 1

# map() changes each text value to its matching number
data["M/F"] = data["M/F"].map({"M": 0, "F": 1})


# 9. Remove Rare Classes

# Count the number of samples in each class
class_counts = data["class"].value_counts()

# Keep classes with more than one sample only
# because SMOTE cannot work with a single sample
valid_classes = class_counts[class_counts > 1].index

# Remove very rare classes from the dataset
# isin() : checks if each row value exists
# inside valid_classes or not.
data = data[data["class"].isin(valid_classes)]

# 4. Define X and y

# Input features used for prediction
features = ["Age", "MMSE", "CDR", "M/F"]
X = data[features]

# Target output (class labels)
# Possible classes:
# NonDemented, Very Mild Demented, Mild Demented
y = data["class"]




# 10. Apply SMOTE

# SMOTE (Synthetic Minority Over-sampling Technique)
# is used to balance the dataset.

# Some classes may have fewer samples than others,
# which can make the model biased toward larger classes.

# random_state=42 keeps the results the same
# every time the code runs.

smote = SMOTE(random_state=42)

# X contains the input features such as:
# Age, MMSE, CDR, and Gender.

# y contains the output classes such as:
# NonDemented, MildDemented,
# and VeryMildDemented.

# fit_resample():
# fit      -> checks the class distribution
# resample -> creates new artificial samples
# for small classes to balance the dataset.

# X_resampled -> new balanced input data
# y_resampled -> new balanced output classes

X_resampled, y_resampled = smote.fit_resample(X, y)


# 11. Scaling

# We use RobustScaler to make the data values more balanced.
# Scaling helps the machine learning model understand the data better.
# RobustScaler is useful because it reduces the effect of very large or unusual values (outliers).

scaler = RobustScaler()

# fit_transform() does two things:
# 1. Learn the data range and statistics
# 2. Transform the data into scaled values

X_scaled = scaler.fit_transform(X_resampled)



# 12. Train-Test Split

# Here we split the dataset into:
# - Training data (80%) → used to teach the model
# - Testing data (20%) → used to test model performance

# random_state=42 makes sure we get the same random split every time

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled,
    test_size=0.2,
    random_state=42
)



# 13. Train Model

# Import Random Forest Classifier
# Random Forest is a machine learning algorithm that uses many decision trees
# and combines their answers to make better predictions.

from sklearn.ensemble import RandomForestClassifier

# Create the model
# random_state=42 keeps results consistent

model = RandomForestClassifier(random_state=42)

# Train the model using the training data
# The model learns patterns from X_train and y_train

model.fit(X_train, y_train)



# 14. Predictions

# After training, we use the model to predict results for the test data

y_pred = model.predict(X_test)



# 15. Evaluation

# Evaluate how good the model is

print("\n==============================")

# Accuracy shows how many predictions were correct
# Example: 0.90 means 90% correct predictions

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")

# Classification Report shows:
# - Precision → how accurate positive predictions are
# - Recall → how many real positives were found
# - F1-score → balance between precision and recall
# - Support → number of samples

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")

# Confusion Matrix compares:
# Actual values VS Predicted values

cm = confusion_matrix(y_test, y_pred)

print(cm)



# 16. Confusion Matrix Plot

# Visualize the confusion matrix using a heatmap
# annot=True → shows the numbers inside the boxes
# fmt="d" → display numbers as integers

plt.figure()

sns.heatmap(cm, annot=True, fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")   # Predicted values by model
plt.ylabel("Actual")      # Real values

plt.show()


# 17. New Patient Prediction

# Create a new patient record for testing the model
# Example values:
# Age → patient's age
# MMSE → Mini-Mental State Examination score
# CDR → Clinical Dementia Rating
# M/F → Gender
#          0 = Male
#          1 = Female

new_patient = pd.DataFrame([{
    "Age": 0,
    "MMSE": 0,
    "CDR": 0,
    "M/F": 0
}])


# Scale the new patient data using the same scaler
# IMPORTANT:
# We must scale new data exactly like training data

new_patient_scaled = scaler.transform(new_patient)


# Predict the result for the new patient
# The model will return the predicted class

prediction = model.predict(new_patient_scaled)

print("\nPrediction for new patient:", prediction[0])


# Print the average MMSE value from the dataset
# mean() calculates the average

print(data['MMSE'].mean())


# Save the trained model and scaler
# This allows us to use them later without retraining

import joblib

# Save machine learning model
joblib.dump(model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ==============================
# 1. Import Libraries
# ==============================
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


# 3. Drop Unnecessary Columns

# Remove columns that are not important for prediction
colums_to_drop=['ID','Hand','Educ','SES','eTIV','nWBV','ASF','Delay']
data=data.drop(columns=colums_to_drop)



# 4. Basic Info

# Display sample rows, dataset information,
# missing values, and class distribution
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["class"].value_counts())


# ==============================
# 4. Visualizations
# ==============================
main_color = "#6A5ACD"

# Class Distribution
sns.countplot(x='class', data=data, color=main_color)
plt.title("Class Distribution")
plt.show()

# MMSE Distribution
sns.histplot(data['MMSE'], kde=True, color=main_color)
plt.title("MMSE Distribution")
plt.show()



# 5. Handling Missing Values

# Fill missing values in MMSE and CDR
# using the mean (average) of each column
data["MMSE"] = data["MMSE"].fillna(data["MMSE"].mean())
data["CDR"] = data["CDR"].fillna(data["CDR"].mean())





# 6. Encoding Gender

# Convert gender values from text to numbers
# M = 0 (Male), F = 1 (Female)
data["M/F"] = data["M/F"].map({"M": 0, "F": 1})


# ==============================
# 7. Remove Rare Classes
# ==============================
class_counts = data["class"].value_counts()
valid_classes = class_counts[class_counts > 1].index
data = data[data["class"].isin(valid_classes)]


# ==============================
# 8. Define X and y
# ==============================
features = ["Age", "MMSE", "CDR",  "M/F"]
X = data[features]
y = data["class"]


# ==============================
# 9. Apply SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nBefore SMOTE:\n", y.value_counts())
print("\nAfter SMOTE:\n", y_resampled.value_counts())




# ==============================
# 10. Scaling
# ==============================
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_resampled)


# ==============================
# 11. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled,
    test_size=0.2,
    random_state=42
)


# ==============================
# 12. Train Model
# ==============================
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ==============================
# 13. Predictions
# ==============================
y_pred = model.predict(X_test)


# ==============================
# 14. Evaluation
# ==============================
print("\n==============================")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ==============================
# 15. Confusion Matrix Plot
# ==============================
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 16. New Patient Prediction
# ==============================

# مثال بيانات مريض جديد (تقدرين تغيرين القيم)
new_patient = pd.DataFrame([{
    "Age": 0,
    "MMSE": 0,
    "CDR": 0,
    "M/F": 0   # 0 = Male, 1 = Female
}])

# نفس الـ scaler المستخدم
new_patient_scaled = scaler.transform(new_patient)

# التنبؤ
prediction = model.predict(new_patient_scaled)

print("\nPrediction for new patient:", prediction[0])

print(data['MMSE'].mean())

import joblib

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")


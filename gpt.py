# ============================================================
# STUDENT ATTENDANCE PREDICTION
# RANDOM FOREST CLASSIFIER
# ============================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Attendance_Prediction_2000.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "student_id",
        "attendance",
        "absence_reason"
    ]
)

y = df["attendance"]


# ============================================================
# 3. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        # Numerical columns
        (
            "num",
            "passthrough",
            numerical_cols
        ),

        # Categorical columns
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

model_pipeline = Pipeline(
    steps=[

        # Preprocessing
        (
            "preprocessor",
            preprocessor
        ),

        # Random Forest Classifier
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        )
    ]
)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining the model...")

model_pipeline.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

# Get trained Random Forest model
rf_model = model_pipeline.named_steps["classifier"]

# Get feature importance
importance = rf_model.feature_importances_

# Get encoded feature names
encoded_features = model_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()


# ============================================================
# CONVERT FEATURE NAMES TO SIMPLE NAMES
# ============================================================

simple_names = []

for feature in encoded_features:

    # Remove num__ and cat__
    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")

    # Convert encoded categorical names
    if "_" in feature:
        original_feature = feature.split("_")[0]

        # Simple names
        if original_feature == "gender":
            feature = "Gender"

        elif original_feature == "course":
            feature = "Course"

        elif original_feature == "year":
            feature = "Year"

        elif original_feature == "parent":
            feature = "Parent Education"

        elif original_feature == "internet":
            feature = "Internet Access"

        elif original_feature == "hostel":
            feature = "Hostel Resident"

        elif original_feature == "class":
            feature = "Class Type"

        elif original_feature == "weather":
            feature = "Weather"

    else:

        # Numerical features
        if feature == "age":
            feature = "Age"

        elif feature == "study_hours":
            feature = "Study Hours"

        elif feature == "sleep_hours":
            feature = "Sleep Hours"

        elif feature == "travel_time_minutes":
            feature = "Travel Time"


    simple_names.append(feature)


# ============================================================
# CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

feature_importance_df = pd.DataFrame({

    "Feature": simple_names,

    "Importance": importance

})


# ============================================================
# COMBINE IMPORTANCE OF SAME ORIGINAL FEATURES
# ============================================================

feature_importance_df = (
    feature_importance_df
    .groupby("Feature", as_index=False)["Importance"]
    .sum()
)


# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(feature_importance_df)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance_df["Feature"][::-1],
    feature_importance_df["Importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.show()


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

y_pred = model_pipeline.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")


print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Accuracy Percentage: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(
    model_pipeline,
    "student_attendance_rf_model.pkl"
)

print("\nModel saved successfully!")

print(
    "File: student_attendance_rf_model.pkl"
)


# ============================================================
# 12. TAKE REAL-WORLD USER INPUT
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    " STUDENT ATTENDANCE PREDICTION"
)

print(
    "=========================================="
)


age = int(
    input("Enter Age: ")
)


gender = input(
    "Enter Gender (male/female): "
).lower()


course = input(
    "Enter Course (bca/bsc/bcom/ba/bba): "
).lower()


year = input(
    "Enter Year (1st year/2nd year/3rd year): "
).lower()


parent_education = input(
    "Enter Parent Education (school/graduate/postgraduate): "
).lower()


internet_access = input(
    "Internet Access (yes/no): "
).lower()


hostel_resident = input(
    "Hostel Resident (yes/no): "
).lower()


class_type = input(
    "Class Type (offline/online): "
).lower()


weather = input(
    "Weather (sunny/cloudy/rainy): "
).lower()


study_hours = float(
    input("Study Hours per day: ")
)


sleep_hours = float(
    input("Sleep Hours per day: ")
)


travel_time_minutes = int(
    input("Travel Time in minutes: ")
)


# ============================================================
# 13. CREATE USER INPUT DATAFRAME
# ============================================================

sample_student = pd.DataFrame(
    [
        {

            "age": age,

            "gender": gender,

            "course": course,

            "year": year,

            "parent_education": parent_education,

            "internet_access": internet_access,

            "hostel_resident": hostel_resident,

            "class_type": class_type,

            "weather": weather,

            "study_hours": study_hours,

            "sleep_hours": sleep_hours,

            "travel_time_minutes": travel_time_minutes

        }
    ]
)


# ============================================================
# 14. PREDICT ATTENDANCE
# ============================================================

prediction = model_pipeline.predict(
    sample_student
)[0]


probability = model_pipeline.predict_proba(
    sample_student
)[0]


# ============================================================
# 15. DISPLAY RESULT
# ============================================================

print("\n")

print(
    "=========================================="
)

print(
    "          PREDICTION RESULT"
)

print(
    "=========================================="
)


print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)


if prediction == 1:

    print(
        "Predicted Attendance: PRESENT"
    )

else:

    print(
        "Predicted Attendance: ABSENT"
    )


print(
    f"Absent Probability: {probability[0] * 100:.2f}%"
)


print(
    f"Present Probability: {probability[1] * 100:.2f}%"
)


print(
    "=========================================="
)
# Patient No-Show Prediction

This project focuses on predicting whether a patient will show up for their scheduled appointment or not, based on historical data. The main goal is to address the issue of patient **no-shows**, which can cause operational inefficiencies in healthcare settings.

## Project Overview

The dataset contains information about patients' appointments, including demographic details, appointment schedules, and whether they attended the appointment. The target variable is the **no-show status** (`Yes`/`No`), where `Yes` means the patient did not show up and `No` means they attended the appointment.

## Table of Contents

- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
  - [XGBoost](#xgboost)
  - [Hyperparameter Tuning]
  - [Handling Imbalanced Data]
- [Key Takeaways](#key-takeaways)

## Data Preprocessing

The first step was to clean and preprocess the data:

1. **Removed invalid records**: Some records had **negative lead times** (e.g., -1, -6), which were logically invalid since appointments can't be scheduled after the appointment date.
2. **Addressed outliers**: We detected **outliers** in the **gap** between the scheduled and appointment days. Using the **log1p transformation**, these outliers were handled to ensure the model's stability.
3. **Dealt with categorical variables**: The `Neighbourhood` column was transformed into its respective **mean values** to prevent data leakage and maintain model integrity. This transformation was done **after the train-test split** to ensure proper handling of the data.

## Feature Engineering

### Key Features

The features influencing patient no-shows were:
- **Lead Time**: The gap between when the appointment is scheduled and the appointment day.
- **Patient Age**: Older patients showed different patterns compared to younger patients.
- **SMS Reminder**: Whether the patient received an SMS reminder prior to the appointment.
- **Health Conditions**: Conditions like **hypertension** and **diabetes** affected attendance.
- **Scholarship Enrollment**: Whether the patient was part of a scholarship program.

## Modeling

### **Data Balancing**

#### SMOTE

Given the significant imbalance between the two classes (80% "show" and 20% "no-show"), I used the SMOTE method to oversample the minority class (no-show) to balance the dataset. This improved the model's ability to predict the minority class.

###  **Model Training and Evaluation**

#### Models Used

I tested multiple models, including:
- **Random Forest Classifier**
- **Logistic Regression**
- **XGBoost Classifier**

Out of these, XGBoost performed the best with an accuracy of 76% and an AUC of 0.72.

#### Hyperparameter Tuning

I used GridSearchCV to tune the hyperparameters of the XGBoost model. After hyperparameter tuning, the accuracy remained at 76%, with the following best parameters:

```python
Best Parameters: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200, 'subsample': 0.8}
```

#### Downsampling
To further address the class imbalance, I performed downsampling to match the number of "no-show" and "show" patients. After downsampling, the accuracy dropped to 68%, but the model's ability to predict "yes" (no-show) improved significantly.

```python
Accuracy: 0.68
AUC Score: 0.73
```

## Key Takeaways
#### Class Imbalance: 

The dataset had a significant imbalance between "show" and "no-show" patients. Using SMOTE helped to balance the dataset and improve model performance.

#### XGBoost: 
XGBoost performed the best with an accuracy of 76%, although hyperparameter tuning did not result in further improvement.

#### Downsampling: 
Downsampling reduced the overall accuracy but improved the model's ability to predict no-shows, which could be crucial depending on the business objective.

## Conclusion
This project successfully demonstrated the challenges of predicting rare events in an imbalanced dataset and the impact of various techniques like SMOTE, hyperparameter tuning, and downsampling. The final model, XGBoost, is well-suited for identifying no-shows, with a good balance of accuracy and recall for the minority class.


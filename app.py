import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Employee Dataset
employee_data = pd.DataFrame({
    'emp_id': ['HR8270', 'TECH1860'],
    'age': [28, 50],
    'Dept': ['HR', 'Technology'],
    'location': ['Suburb', 'Suburb'],
    'education': ['PG', 'PG'],
    'recruitment_type': ['Referral', 'Walk-in'],
    'job_level': [5, 3],
    'rating': [2, 5],
    'onsite': [0, 1],
    'awards': [1, 2],
    'certifications': [0, 1],
    'salary': [86750, 42419],
    'satisfied': [1, 0]
})

# 2. Load Sprint Dataset
sprint_data = pd.DataFrame({
    'sprintId': [7],
    'sprintName': ['Q2 Sprint 1'],
    'sprintState': ['CLOSED'],
    'sprintStartDate': ['12-May-14'],
    'sprintEndDate': ['17-May-14'],
    'sprintCompleteDate': ['19-May-14'],
    'total': [24],
    'completedIssuesCount': [9],
    'issuesNotCompletedInCurrentSprint': [9],
    'puntedIssues': [3],
    'issuesCompletedInAnotherSprint': [0],
    'issueKeysAddedDuringSprint': [5],
    'completedIssuesInitialEstimateSum': [0],
    'completedIssuesEstimateSum': [0],
    'puntedIssuesInitialEstimateSum': [0],
    'puntedIssuesEstimateSum': [0],
    'issuesNotCompletedInitialEstimateSum': [0],
    'issuesNotCompletedEstimateSum': [0],
    'issuesCompletedInAnotherSprintInitialEstimateSum': [0],
    'issuesCompletedInAnotherSprintEstimateSum': [0],
    'NoOfDevelopers': [6],
    'SprintLength': [7]
})

# 3. Preprocessing - Encode categorical columns
label_encoder = LabelEncoder()
employee_data['Dept'] = label_encoder.fit_transform(employee_data['Dept'])
employee_data['location'] = label_encoder.fit_transform(employee_data['location'])
employee_data['education'] = label_encoder.fit_transform(employee_data['education'])
employee_data['recruitment_type'] = label_encoder.fit_transform(employee_data['recruitment_type'])

# 4. Feature Engineering
# Define success rate of the sprint as completed issues over total issues
sprint_data['success_rate'] = sprint_data['completedIssuesCount'] / sprint_data['total']

# 5. Merge Datasets based on sprint or emp_id
# For simplicity, we'll assume emp_id can be used to relate employees to sprints
# In real cases, you might need to join on developer allocation or specific mapping
combined_data = employee_data.join(sprint_data)

# Drop unnecessary columns for model training
combined_data = combined_data.drop(columns=['emp_id', 'sprintId', 'sprintName', 'sprintState',
                                            'sprintStartDate', 'sprintEndDate', 'sprintCompleteDate'])

# 6. Splitting the dataset into features (X) and target (y)
X = combined_data.drop(columns=['success_rate'])
y = combined_data['success_rate']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Train the Model (Random Forest Regressor for prediction)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 10. Predict and Evaluate the Model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 11. Insights from the Model
importances = model.feature_importances_
feature_names = X.columns
feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Use model to provide insights on sprint success (example for new data)
new_employee_data = pd.DataFrame({
    'age': [35],
    'Dept': [1],  # Encoded 'Technology'
    'location': [1],  # Encoded 'Suburb'
    'education': [1],  # Encoded 'PG'
    'recruitment_type': [1],  # Encoded 'Walk-in'
    'job_level': [4],
    'rating': [4],
    'onsite': [1],
    'awards': [1],
    'certifications': [1],
    'salary': [60000],
    'satisfied': [1]
})

new_sprint_data = pd.DataFrame({
    'total': [30],
    'completedIssuesCount': [20],
    'issuesNotCompletedInCurrentSprint': [5],
    'puntedIssues': [3],
    'issuesCompletedInAnotherSprint': [1],
    'issueKeysAddedDuringSprint': [7],
    'completedIssuesInitialEstimateSum': [0],
    'completedIssuesEstimateSum': [0],
    'puntedIssuesInitialEstimateSum': [0],
    'puntedIssuesEstimateSum': [0],
    'issuesNotCompletedInitialEstimateSum': [0],
    'issuesNotCompletedEstimateSum': [0],
    'issuesCompletedInAnotherSprintInitialEstimateSum': [0],
    'issuesCompletedInAnotherSprintEstimateSum': [0],
    'NoOfDevelopers': [8],
    'SprintLength': [10]
})

# Combine new data for predictions
new_combined_data = new_employee_data.join(new_sprint_data)
new_combined_data_scaled = scaler.transform(new_combined_data)
predicted_success_rate = model.predict(new_combined_data_scaled)
print(f"Predicted Success Rate for New Sprint: {predicted_success_rate[0]}")

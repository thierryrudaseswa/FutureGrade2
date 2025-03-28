import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

np.random.seed(42)
n_samples = 1000

# Generate subject scores (0-100) basically
subjects = ['mathematics', 'english', 'science', 'history', 'geography']
data = {}
for subject in subjects:
    for term in ['term1', 'term2', 'term3']:
        col_name = f'{subject}_{term}'
        data[col_name] = np.random.normal(75, 15, n_samples)
        data[col_name] = np.clip(data[col_name], 0, 100)

# Generation of  attendance hours (0-3) beyond three it shows excellence
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
for day in days:
    col_name = f'{day}_hours'
    data[col_name] = np.random.normal(2, 0.5, n_samples)
    data[col_name] = np.clip(data[col_name], 0, 3)

# Generate assignment completion rate (0-100)
data['assignment_completion_rate'] = np.random.normal(85, 10, n_samples)
data['assignment_completion_rate'] = np.clip(data['assignment_completion_rate'], 0, 100)

# Calculate national exam score based on three to work on
weights = {
    'subject_weight': 0.90,  # 90% weight for subject scores
    'attendance_weight': 0.05,  # 5% weight for attendance
    'assignment_weight': 0.05   # 5% weight for assignments
}

# Calculation on subject score contribution
subject_scores = np.zeros(n_samples)
for subject in subjects:
    for term in ['term1', 'term2', 'term3']:
        subject_scores += data[f'{subject}_{term}']
subject_scores = subject_scores / (len(subjects) * 3)

# Calculation on attendance contribution
attendance_scores = np.zeros(n_samples)
for day in days:
    attendance_scores += data[f'{day}_hours']
attendance_scores = attendance_scores / len(days)

# Calculate final score
data['national_exam_score'] = (
    weights['subject_weight'] * subject_scores +
    weights['attendance_weight'] * (attendance_scores / 3 * 100) +
    weights['assignment_weight'] * data['assignment_completion_rate']
)


df = pd.DataFrame(data)

df.to_csv('training_data.csv', index=False)

X = df.drop('national_exam_score', axis=1)
y = df['national_exam_score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=2000, 
    max_depth=30,     
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")


model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.joblib')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")


sample_data = {
    'perfect_student': {
        'mathematics_term1': 100, 'mathematics_term2': 100, 'mathematics_term3': 100,
        'english_term1': 100, 'english_term2': 100, 'english_term3': 100,
        'science_term1': 100, 'science_term2': 100, 'science_term3': 100,
        'history_term1': 100, 'history_term2': 100, 'history_term3': 100,
        'geography_term1': 100, 'geography_term2': 100, 'geography_term3': 100,
        'monday_hours': 3, 'tuesday_hours': 3, 'wednesday_hours': 3,
        'thursday_hours': 3, 'friday_hours': 3, 'saturday_hours': 3,
        'sunday_hours': 3, 'assignment_completion_rate': 100
    },
    'excellent_student': {
        'mathematics_term1': 90, 'mathematics_term2': 95, 'mathematics_term3': 92,
        'english_term1': 92, 'english_term2': 90, 'english_term3': 95,
        'science_term1': 95, 'science_term2': 92, 'science_term3': 90,
        'history_term1': 90, 'history_term2': 95, 'history_term3': 92,
        'geography_term1': 92, 'geography_term2': 90, 'geography_term3': 95,
        'monday_hours': 2.5, 'tuesday_hours': 2.5, 'wednesday_hours': 2.5,
        'thursday_hours': 2.5, 'friday_hours': 2.5, 'saturday_hours': 2.5,
        'sunday_hours': 2.5, 'assignment_completion_rate': 95
    },
    'good_student': {
        'mathematics_term1': 80, 'mathematics_term2': 85, 'mathematics_term3': 82,
        'english_term1': 82, 'english_term2': 80, 'english_term3': 85,
        'science_term1': 85, 'science_term2': 82, 'science_term3': 80,
        'history_term1': 80, 'history_term2': 85, 'history_term3': 82,
        'geography_term1': 82, 'geography_term2': 80, 'geography_term3': 85,
        'monday_hours': 2, 'tuesday_hours': 2, 'wednesday_hours': 2,
        'thursday_hours': 2, 'friday_hours': 2, 'saturday_hours': 2,
        'sunday_hours': 2, 'assignment_completion_rate': 85
    }
}

print("\nSample Predictions:")
for student_type, data in sample_data.items():
    X_sample = np.array(list(data.values())).reshape(1, -1)
    X_sample_scaled = scaler.transform(X_sample)
    prediction = model.predict(X_sample_scaled)[0]
    print(f"{student_type.replace('_', ' ').title()}: {prediction:.2f} points") 
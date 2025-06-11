import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_data(n_samples=1000):
    """Generate student performance data."""
    # Generate features
    hours_studied = np.random.normal(6, 2, n_samples)  # Mean 6 hours, std 2
    sleep_hours = np.random.normal(7, 1, n_samples)    # Mean 7 hours, std 1
    attendance = np.random.uniform(0.6, 1.0, n_samples)  # 60-100% attendance
    previous_grades = np.random.normal(75, 10, n_samples)  # Mean 75, std 10
    
    # Generate target variables
    # Final score depends on all features with some noise
    final_score = (
        0.3 * hours_studied * 10 +  # Hours studied impact
        0.2 * sleep_hours * 10 +    # Sleep impact
        0.3 * attendance * 100 +    # Attendance impact
        0.2 * previous_grades +     # Previous grades impact
        np.random.normal(0, 5, n_samples)  # Random noise
    )
    
    # Clip final score between 0 and 100
    final_score = np.clip(final_score, 0, 100)
    
    # Create pass/fail classification (pass if score >= 60)
    pass_fail = (final_score >= 60).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'hours_studied': hours_studied,
        'sleep_hours': sleep_hours,
        'attendance': attendance,
        'previous_grades': previous_grades,
        'final_score': final_score,
        'pass_fail': pass_fail
    })
    
    return data

def train_models(data):
    """Train regression and classification models."""
    # Prepare features and targets
    X = data[['hours_studied', 'sleep_hours', 'attendance', 'previous_grades']]
    y_reg = data['final_score']
    y_clf = data['pass_fail']
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    
    # Train regression model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_reg_train)
    reg_pred = reg_model.predict(X_test)
    reg_mse = mean_squared_error(y_reg_test, reg_pred)
    
    # Train classification model
    clf_model = DecisionTreeClassifier(random_state=42)
    clf_model.fit(X_train, y_clf_train)
    clf_pred = clf_model.predict(X_test)
    clf_accuracy = accuracy_score(y_clf_test, clf_pred)
    
    return reg_model, clf_model, reg_mse, clf_accuracy

def plot_correlation_heatmap(data):
    """Plot correlation heatmap of features."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def get_user_input():
    """Get student performance data from user input."""
    print("\n=== Student Performance Prediction ===")
    print("Please enter the following information:")
    
    # Get hours studied
    while True:
        try:
            hours_studied = float(input("Hours studied per day (0-24): "))
            if 0 <= hours_studied <= 24:
                break
            else:
                print("Please enter a value between 0 and 24.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get sleep hours
    while True:
        try:
            sleep_hours = float(input("Sleep hours per day (0-24): "))
            if 0 <= sleep_hours <= 24:
                break
            else:
                print("Please enter a value between 0 and 24.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get attendance
    while True:
        try:
            attendance = float(input("Attendance percentage (0-100): ")) / 100
            if 0 <= attendance <= 1:
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get previous grades
    while True:
        try:
            previous_grades = float(input("Previous grades (0-100): "))
            if 0 <= previous_grades <= 100:
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    return pd.DataFrame({
        'hours_studied': [hours_studied],
        'sleep_hours': [sleep_hours],
        'attendance': [attendance],
        'previous_grades': [previous_grades]
    })

def main():
    # Generate data
    print("Generating synthetic student data...")
    data = generate_student_data()
    
    # Plot correlation heatmap
    print("Generating correlation heatmap...")
    plot_correlation_heatmap(data)
    
    # Train models
    print("Training models...")
    reg_model, clf_model, reg_mse, clf_accuracy = train_models(data)
    
    # Print model performance
    print("\nModel Performance:")
    print(f"Regression MSE: {reg_mse:.2f}")
    print(f"Classification Accuracy: {clf_accuracy:.2%}")
    
    # Get user input for prediction
    while True:
        # Get student data from user
        student_data = get_user_input()
        
        # Make predictions
        predicted_score = reg_model.predict(student_data)[0]
        predicted_pass = clf_model.predict(student_data)[0]
        
        # Print predictions
        print("\n=== Prediction Results ===")
        print(f"Hours Studied: {student_data['hours_studied'][0]:.1f}")
        print(f"Sleep Hours: {student_data['sleep_hours'][0]:.1f}")
        print(f"Attendance: {student_data['attendance'][0]:.1%}")
        print(f"Previous Grades: {student_data['previous_grades'][0]:.1f}")
        print(f"Predicted Final Score: {predicted_score:.1f}")
        print(f"Predicted Pass/Fail: {'Pass' if predicted_pass == 1 else 'Fail'}")
        
        # Ask if user wants to make another prediction
        another = input("\nWould you like to make another prediction? (y/n): ").lower()
        if another != 'y':
            break
    
    print("\nThank you for using the Student Performance Predictor!")

if __name__ == "__main__":
    main() 

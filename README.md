# 🎓 Student Performance Predictor

A beginner-friendly machine learning project that predicts student performance based on input features like hours studied, sleep hours, attendance, and previous grades. It includes both **regression** (to predict final score) and **classification** (to predict pass/fail).

## 📌 Features

- Predict student final exam score using Linear Regression
- Predict pass/fail status using Decision Tree Classification
- Correlation heatmap visualization of features
- Interactive user input for predictions
- Easy to understand and modify

## 🧠 Machine Learning Concepts Used

- **Linear Regression** for predicting final scores  
- **Decision Tree Classifier** for predicting pass/fail  
- **Train-test split**, **Mean Squared Error**, and **Accuracy Score** for evaluation  
- **Seaborn heatmap** for feature correlation

## 🚀 Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python student_performance_predictor.py
   ```

## 📊 Project Structure

- `student_performance_predictor.py`: Main Python script containing all the code
- `requirements.txt`: List of Python dependencies
- `correlation_heatmap.png`: Generated visualization of feature correlations

## 🔍 How It Works

1. The script generates synthetic student data with the following features:
   - Hours studied per day
   - Sleep hours per day
   - Attendance percentage
   - Previous grades

2. It trains two models:
   - A Linear Regression model to predict the final score
   - A Decision Tree Classifier to predict pass/fail status

3. The script generates a correlation heatmap to visualize relationships between features

4. The user can then input student data to get predictions:
   - Hours studied per day (0-24)
   - Sleep hours per day (0-24)
   - Attendance percentage (0-100)
   - Previous grades (0-100)

5. The models predict:
   - The final exam score (regression)
   - Whether the student will pass or fail (classification)

## 📝 Example Output

The script will show:
- Model performance metrics (MSE for regression, accuracy for classification)
- Feature correlation heatmap
- Interactive prompts for user input
- Prediction results for the entered student data

## 🛠️ Customization

You can modify the script to:
- Change the number of samples generated
- Adjust feature weights in the data generation
- Modify the pass/fail threshold (currently set at 60)
- Add new features or models
- Change the input validation ranges

## 📚 Dependencies

- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

## 📄 License

This project is open source and available under the MIT License. 
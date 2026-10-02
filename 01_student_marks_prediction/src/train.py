# Import Pandas
import pandas as pd

# Load the student dataset
df = pd.read_csv("C:\\Users\\HP\\Python Programs\\Machine Learning\\01_student_marks_prediction\\data\\students.csv")

# Select the input feature
X = df[["Study_Hours"]]

# Select the target variable
y = df["Marks"]

# Display the input features
print("Features:")
print(X)

# Display the target
print("Target:")
print(y)

# Import the train_test_split function
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display the sizes of the datasets
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

# Import Linear Regression
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Display the learned coefficient
print("Coefficient:", model.coef_[0])

# Display the learned intercept
print("Intercept:", model.intercept_)

# Make predictions using test features
y_pred = model.predict(X_test)

# Display actual and predicted marks
print("Actual marks:", y_test.to_list())
print("Predicted marks:", y_pred)

# Import evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate RMSE
rmse = mse ** 0.5

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

# Display the results
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)

# Import Matplotlib
import matplotlib.pyplot as plt

# Plot actual student records
plt.scatter(
    X["Study_Hours"],
    y,
    label="Actual data"
)

# Plot the regression line
plt.plot(
    X["Study_Hours"],
    model.predict(X),
    color="red",
    label="Regression line"
)

# Add graph title and axis labels
plt.title("Student Marks Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

# Display legend and grid
plt.legend()
plt.grid(True)

# Display the graph
plt.show()
# Create a new student input
new_student = pd.DataFrame({
    "Study_Hours": [7.5]
})

# Predict marks
predicted_marks = model.predict(new_student)

# Display the prediction
print("Predicted marks:", predicted_marks[0])
# Import Joblib
import joblib

# Save the trained model
joblib.dump(model, "C:\\Users\\HP\\Python Programs\\Machine Learning\\01_student_marks_prediction\\models\\student_marks_model.pkl")

print("Model saved successfully!")
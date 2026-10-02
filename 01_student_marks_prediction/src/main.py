# Import the Pandas library
import pandas as pd

# Create a dictionary containing student data
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 50, 55, 65, 70, 78, 85, 88, 95]
}

# Convert the dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Display the complete dataset
print(df)

# Save the dataset as a CSV file
df.to_csv("C:\\Users\\HP\\Python Programs\\Machine Learning\\01_student_marks_prediction\\data\\students.csv", index=False)

# Display a success message
print("Dataset created successfully!")

# Display the first five rows
print("\nFirst five rows:")
print(df.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display column names and data types
print("\nDataset information:")
df.info()

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Import Matplotlib
import matplotlib.pyplot as plt

# Create a scatter plot
plt.scatter(df["Study_Hours"], df["Marks"])

# Add the graph title
plt.title("Study Hours vs Exam Marks")

# Label the horizontal axis
plt.xlabel("Study Hours")

# Label the vertical axis
plt.ylabel("Exam Marks")

# Display a grid
plt.grid(True)

# Display the graph
plt.show()
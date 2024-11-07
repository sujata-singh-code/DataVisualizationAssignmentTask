# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset with YearsExperience and Salary

data = data = {
    "YearsExperience": [
        1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.5, 4.9,
        5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5
    ],
    "Salary": [
        39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 
        63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 
        91738, 98273, 101302, 113812, 105582, 116969, 112635, 122391
    ]
} 
data = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = data[['YearsExperience']]
y = data['Salary']

# Spliting Mt dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fit the linear regression model
model = LinearRegression()

# Plotting the results
plt.figure(figsize=(10, 6))
# Scatter plot for the actual data
sns.scatterplot(x=data['YearsExperience'], y=data['Salary'], color='blue', label='Actual data')

#new functionality
plt.title("Salary vs. Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

# Show ploting
plt.show()

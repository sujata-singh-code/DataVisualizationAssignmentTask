from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Datase
iris = load_iris()
X = iris.data       
y = iris.target   

# Spliting the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Printing Sample Sets of Data
print("Number of samples in the training set:", X_train.shape[0])
print("Number of samples in the testing set:", X_test.shape[0])

# Convert target values to species names for better readability
species_names = [iris.target_names[label] for label in y]
species_train = [iris.target_names[label] for label in y_train]
species_test = [iris.target_names[label] for label in y_test]

# Create a DataFrame for visualizing the data split
data_split_df = pd.DataFrame({
    'Set': ['Train'] * len(y_train) + ['Test'] * len(y_test),
    'Species': species_train + species_test
})

# Setcolor
palette = sns.color_palette("Set2", len(iris.target_names))

#Using Bar Chart for Count of Each Species in Training and Test Sets
plt.figure(figsize=(10, 6))
sns.countplot(x='Set', hue='Species', data=data_split_df, palette=palette)
plt.title("Distribution of Species in Training and Testing Sets")
plt.xlabel("Data Split")
plt.ylabel("Count")
plt.legend(title="Species")
plt.show()

#Using Pie Chart for Overall Data Split (Train vs Test)
split_counts = [len(y_train), len(y_test)]
split_labels = ['Train', 'Test']
plt.figure(figsize=(8, 6))
plt.pie(split_counts, labels=split_labels, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
plt.title("Overall Train-Test Split (80-20)")
plt.show()

#Using Pie Charts for Species Distribution in Train and Test Sets
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
train_counts = data_split_df[data_split_df['Set'] == 'Train']['Species'].value_counts()
test_counts = data_split_df[data_split_df['Set'] == 'Test']['Species'].value_counts()

# Training Set Pie Chart
ax[0].pie(train_counts, labels=train_counts.index, autopct='%1.1f%%', startangle=90, colors=palette)
ax[0].set_title("Species Distribution in Training Set")

# Testing Set Pie Chart
ax[1].pie(test_counts, labels=test_counts.index, autopct='%1.1f%%', startangle=90, colors=palette)
ax[1].set_title("Species Distribution in Testing Set")

plt.show()


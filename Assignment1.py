import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Loading Dataset
iris = load_iris()

# Creating my DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Displaying the first five rows
print("First five rows:")
print(iris_df.head())

# Display the shape of the dataset
print("Dataset shape:")
print(iris_df.shape)

# Display summary 
print("\nSummary statistics:")
print(iris_df.describe())

#using charts for visualizations
sns.set(style='whitegrid')

# Pairing of Plots
plt.figure(figsize=(10, 6))
sns.pairplot(iris_df, hue='species', palette='Set1', markers=['o', 's', 'D'])
plt.title('Pair Plot of Iris Dataset')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal length (cm)', data=iris_df)
plt.title('Box Plot of Sepal Length by Species')
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal length (cm)', data=iris_df, inner='quartile', palette='Set1')
plt.title('Violin Plot of Petal Length by Species')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=iris_df, style='species', palette='Set1')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()

# Histogram for Sepal Length
plt.figure(figsize=(10, 6))
sns.histplot(data=iris_df, x='sepal length (cm)', hue='species', multiple='stack', bins=10, kde=True)
plt.title('Histogram of Sepal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Counting for Plot for Species
plt.figure(figsize=(10, 6))
sns.countplot(x='species', data=iris_df, palette='Set1')
plt.title('Count of Each Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
plt.show()

# Heatmap of Cor-relation 
plt.figure(figsize=(10, 6))
correlation_matrix = iris_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Cor-relation ')
plt.show()

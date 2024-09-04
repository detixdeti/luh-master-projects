import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv("data/exams.csv")

# Function to plot math scores by gender
def plot_gender(gender_data, gender_name, color):
    plt.figure(figsize=(10, 6))
    plt.hist(gender_data['math score'], bins=20, color=color, alpha=0.7)
    plt.title(f'Math Scores of {gender_name.capitalize()} Students')
    plt.xlabel('Math Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Separate the data by gender
female_students = df[df['gender'] == 'female']
male_students = df[df['gender'] == 'male']

# Plotting
plot_gender(female_students, 'female', 'pink')
plot_gender(male_students, 'male', 'blue')

# Clustering function
def clustering(data):
    # Select relevant features for clustering: math score, reading score, writing score
    features = data[['math score', 'reading score', 'writing score']]

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(features)

    # Plotting the clusters with respect to race/ethnicity
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='math score', y='reading score', hue='cluster', style='race/ethnicity', data=data, palette='Set1', s=100)
    plt.title('Clustering of Grade Performances with Respect to Race/Ethnicity')
    plt.xlabel('Math Score')
    plt.ylabel('Reading Score')
    plt.grid(True)
    plt.show()

# Call the clustering function
clustering(df)

mean_ethnicity = df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].mean()
median_ethnicity = df.groupby('race/ethnicity')[['math score', 'reading score', 'writing score']].median()

print(mean_ethnicity)
print(median_ethnicity)

sns.boxplot(x='race/ethnicity', y='math score', data=df)
sns.boxplot(x='race/ethnicity', y='reading score', data=df)
sns.boxplot(x='race/ethnicity', y='writing score', data=df)
plt.show()

math_scores = [df[df['race/ethnicity'] == group]['math score'] for group in df['race/ethnicity'].unique()]
reading_scores = [df[df['race/ethnicity'] == group]['reading score'] for group in df['race/ethnicity'].unique()]
writing_scores = [df[df['race/ethnicity'] == group]['writing score'] for group in df['race/ethnicity'].unique()]

f_stat_math, p_value_math = f_oneway(*math_scores)
f_stat_reading, p_value_reading = f_oneway(*reading_scores)
f_stat_writing, p_value_writing = f_oneway(*writing_scores)

print("ANOVA Results for Math: F-statistic = {}, p-value = {}".format(f_stat_math, p_value_math))
print("ANOVA Results for Reading: F-statistic = {}, p-value = {}".format(f_stat_reading, p_value_reading))
print("ANOVA Results for Writing: F-statistic = {}, p-value = {}".format(f_stat_writing, p_value_writing))


X = pd.get_dummies(df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']], drop_first=True)
y = df[['math score', 'reading score', 'writing score']]

model = LinearRegression()
model.fit(X, y)

# Predicting scores for new data
new_data = pd.get_dummies(df, drop_first=True)
predictions = model.predict(new_data)
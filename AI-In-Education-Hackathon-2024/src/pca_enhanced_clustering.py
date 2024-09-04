import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('data/exams.csv')

# Select relevant features for clustering
features = pd.get_dummies(df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']], drop_first=True)

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# Plotting the clusters with PCA-reduced data
plt.figure(figsize=(12, 8))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=df['cluster'], style=df['race/ethnicity'], palette='Set1', s=100)
plt.title('Enhanced Clustering with PCA-Reduced Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

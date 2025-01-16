# Customer Segmentation Project

This project demonstrates customer segmentation using the K-means clustering algorithm on the Mall Customers dataset.

## Short Description

The project focuses on customer segmentation by applying K-means clustering on a dataset of mall customers. It includes data preprocessing, visualization, and evaluation of clustering results.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/customer-segmentation.git
    cd customer-segmentation
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")
Downloading and Loading Dataset
python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
print("Path to dataset files:", path)

# Loading dataset
file_path = os.path.join(path, "Mall_Customers.csv")
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(df.head())
    print(df.describe())
else:
    print(f"Error: File not found at {file_path}.")

Visualizing Data
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], color='green')
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
------------------------------------------------------------
Visualizing Data
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], color='green')
plt.title('Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
----------------------------------------------------------
Data Preprocessing and K-means Clustering
X = df.iloc[:, [3, 4]].values  # Selecting the features for clustering

# Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
-------------------------------------------------------
K-means Clustering
# Applying K-means to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
-------------------------------------------------------
Visualizing the Clusters

# Visualizing the clusters
for cluster in range(5):  
    plt.scatter(
        X[y_kmeans == cluster, 0], 
        X[y_kmeans == cluster, 1], 
        s=100, 
        label=f'Cluster {cluster + 1}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1], 
    s=300, 
    c='yellow', 
    label='Centroids'
)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt

---------------------------------------------------



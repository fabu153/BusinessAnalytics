# Business Data Clustering Dashboard

## Overview
The **Business Data Clustering Dashboard** is an interactive Streamlit application designed to help users cluster and analyze business datasets. It offers robust preprocessing, clustering algorithms, and visualization capabilities to generate meaningful insights.

## Features
- **Data Upload**: Upload CSV files with numeric and categorical data for analysis.
- **Preprocessing**:
  - Handles missing values (mean for numeric, mode for categorical).
  - Scales numeric features with StandardScaler.
  - Encodes categorical features using OneHotEncoder.
  - Removes unecessary columns (Eg. ID columns)
- **Clustering Algorithms**:
  - **KMeans**: Automatically suggests the optimal number of clusters using the silhouette score.
  - **DBSCAN**: Recommends an optimal epsilon value based on the k-distance plot.
- **Visualization**:
  - Cluster distribution bar chart.
  - PCA-based 2D cluster visualization.
  - t-SNE-based 2D cluster visualization.
- **Cluster Insights**:
  - Cluster-level summaries and descriptions.
  - Top feature contributions for each cluster.
- **Downloadable Outputs**:
  - Export cluster insights and descriptions as CSV files.

## Installation
- Clone repo
- cd BusinessAnalytics
- Streamlit run main.py

### Prerequisites
- Python 3.8 or above
- pip + requirements.txt

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/fabu153/BusinessAnalytics.git
   cd BusinessAnalytics

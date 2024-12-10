import os
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize Upload Directory
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def main():
    """
    Main function to run the Streamlit dashboard for business data clustering.
    
    Initializes session state variables, handles file upload, selects clustering algorithms, 
    and displays visualizations for clustered data.
    """
    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "selected_visualization" not in st.session_state:
        st.session_state.selected_visualization = "None"
    if "cluster_insights" not in st.session_state:
        st.session_state.cluster_insights = None
    if "cluster_descriptions" not in st.session_state:
        st.session_state.cluster_descriptions = None

    # App Title
    st.title("Business Data Clustering Dashboard")

    # Help Section
    with st.expander("Help"):
        st.write("""
        - **Silhouette Score**: Measures the quality of clusters (higher is better).
        - **PCA Visualization**: Displays clusters based on the first two principal components.
        - **t-SNE Visualization**: Displays clusters in a reduced dimensional space for interpretability.
        - **Top Features**: Identifies the features most impactful for clustering.
        """)

    # File Upload Section
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        # Save uploaded file and preprocess the data
        file_path = save_uploaded_file(uploaded_file)
        preprocessed_data, scaler = preprocess_csv(file_path)

        # Display Preprocessed Data
        st.write("Preprocessed Data Sample:")
        st.dataframe(preprocessed_data.head())

        # Clustering Algorithm Selection
        algorithm = st.selectbox("Choose Clustering Algorithm", ["KMeans", "DBSCAN"])
        
        # Handle selected clustering algorithm
        if algorithm == "KMeans":
            handle_kmeans(preprocessed_data, scaler)
        elif algorithm == "DBSCAN":
            handle_dbscan(preprocessed_data, scaler)

        # Visualization Section
        handle_visualizations(preprocessed_data)

def save_uploaded_file(uploaded_file):
    """
    Saves the uploaded file to the server.
    
    Args:
        uploaded_file: The file object uploaded by the user.
    
    Returns:
        str: The path to the saved file.
    """
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def preprocess_csv(file):
    """
    Preprocesses a CSV file for clustering by handling missing values, encoding categorical data, 
    and scaling numeric data.

    Args:
        file (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - preprocessed_data (pd.DataFrame): The cleaned and transformed data.
            - scaler (StandardScaler): Scaler used to standardize the numeric data.
    """
    data = pd.read_csv(file)
    if data.empty:
        raise ValueError("The uploaded file is empty. Please upload a valid CSV file.")
    
    # Remove unnamed and empty columns
    data = data.loc[:, ~data.columns.str.contains(r"^Unnamed$|^$", regex=True)]
    # Drop columns with "id" in their names
    data = data.drop(columns=[col for col in data.columns if "id" in col.lower()], errors="ignore")

    # Separate numeric and categorical data
    numeric_data = data.select_dtypes(include=["number"])
    categorical_data = data.select_dtypes(include=["object", "category", "bool"])

    # Fill missing values
    numeric_data = numeric_data.fillna(numeric_data.mean())
    categorical_data = categorical_data.apply(lambda col: col.fillna(col.mode()[0]))

    # Encode categorical data if present
    if not categorical_data.empty:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_categorical = encoder.fit_transform(categorical_data)
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=encoder.get_feature_names_out(categorical_data.columns),
            index=data.index,
        )
    else:
        encoded_categorical_df = pd.DataFrame(index=data.index)

    # Scale numeric data
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(numeric_data)
    scaled_numeric_df = pd.DataFrame(
        scaled_numeric, columns=numeric_data.columns, index=data.index
    )

    # Combine numeric and encoded categorical data
    preprocessed_data = pd.concat([scaled_numeric_df, encoded_categorical_df], axis=1)
    return preprocessed_data, scaler

def handle_kmeans(preprocessed_data, scaler):
    """
    Handles the KMeans clustering process, including finding the optimal number of clusters, 
    running the clustering algorithm, and displaying results.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed data ready for clustering.
        scaler (StandardScaler): Scaler used for data preprocessing.
    """
    with st.spinner("Finding optimal number of clusters..."):
        optimal_clusters = find_optimal_kmeans(preprocessed_data)
    st.success("Optimal Clusters found!")
    st.write(f"Optimal number of clusters: **{optimal_clusters}**")

    # Allow user to adjust the number of clusters
    n_clusters = st.slider("Number of Clusters", 2, 10, optimal_clusters)
    if st.button("Run KMeans Clustering"):
        try:
            with st.spinner("Running KMeans Clustering..."):
                results = train_and_apply_model_kmeans(preprocessed_data, scaler, n_clusters=n_clusters)
            st.session_state.results = results
            st.session_state.cluster_insights = results["cluster_insights"]
            st.session_state.cluster_descriptions = results["descriptions"]
            display_results(results)
        except Exception as e:
            st.error(f"Error running KMeans clustering: {e}")

def handle_dbscan(preprocessed_data, scaler):
    """
    Handles the DBSCAN clustering process, including recommending optimal epsilon, 
    running the clustering algorithm, and displaying results.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed data ready for clustering.
        scaler (StandardScaler): Scaler used for data preprocessing.
    """
    with st.spinner("Recommending optimal eps for DBSCAN..."):
        recommended_eps = recommend_eps_for_dbscan(preprocessed_data)
    st.success("Optimal epsilon found!")
    st.write(f"Recommended epsilon value: **{recommended_eps:.2f}**")

    # Allow user to adjust epsilon and minimum samples
    eps = st.slider("Epsilon", 0.1, 10.0, recommended_eps)
    min_samples = st.slider("Minimum Samples", 1, 10, 5)
    if st.button("Run DBSCAN Clustering"):
        try:
            with st.spinner("Running DBSCAN Clustering..."):
                results = train_and_apply_model_dbscan(preprocessed_data, scaler, eps=eps, min_samples=min_samples)
            st.session_state.results = results
            st.session_state.cluster_insights = results["cluster_insights"]
            st.session_state.cluster_descriptions = results["descriptions"]
            display_results(results)
        except Exception as e:
            st.error(f"Error running DBSCAN clustering: {e}")

def find_optimal_kmeans(data):
    """
    Finds the optimal number of clusters for KMeans clustering using the silhouette score.

    Args:
        data (pd.DataFrame): Preprocessed data ready for clustering.

    Returns:
        int: Optimal number of clusters.
    """
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, 11)
    progress = st.progress(0)

    # Evaluate each number of clusters
    for i, n_clusters in enumerate(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        progress.progress((i + 1) / len(cluster_range))

    return cluster_range[silhouette_scores.index(max(silhouette_scores))]

def recommend_eps_for_dbscan(data, min_samples=5):
    """
    Recommends the optimal epsilon value for DBSCAN clustering using the k-distance graph.

    Args:
        data (pd.DataFrame): Preprocessed data ready for clustering.
        min_samples (int): Minimum samples parameter for DBSCAN. Default is 5.

    Returns:
        float: Recommended epsilon value.
    """
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    k_distances = np.sort(distances[:, -1])
    return k_distances[int(len(k_distances) * 0.9)]

def train_and_apply_model_kmeans(preprocessed_data, scaler, n_clusters):
    """
    Train and apply the KMeans clustering algorithm to the preprocessed data.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed data ready for clustering.
        scaler (StandardScaler): Scaler used to standardize numeric data.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        dict: A dictionary containing:
            - cluster_labels: Labels assigned to each data point.
            - cluster_insights: Mean values for each feature per cluster.
            - descriptions: Descriptive summary of each cluster.
            - metrics: Clustering metrics like silhouette score.
            - top_features: Top features identified by PCA.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preprocessed_data)
    silhouette = silhouette_score(preprocessed_data, labels)
    insights = generate_cluster_insights(preprocessed_data, labels)
    descriptions = generate_cluster_descriptions(insights, scaler)
    top_features = get_top_features_pca(preprocessed_data)
    return {
        "cluster_labels": labels,
        "cluster_insights": insights,
        "descriptions": descriptions,
        "metrics": {"silhouette_score": silhouette},
        "top_features": top_features
    }

def train_and_apply_model_dbscan(preprocessed_data, scaler, eps, min_samples):
    """
    Train and apply the DBSCAN clustering algorithm to the preprocessed data.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed data ready for clustering.
        scaler (StandardScaler): Scaler used to standardize numeric data.
        eps (float): Epsilon parameter for DBSCAN.
        min_samples (int): Minimum samples parameter for DBSCAN.

    Returns:
        dict: A dictionary containing:
            - cluster_labels: Labels assigned to each data point.
            - cluster_insights: Mean values for each feature per cluster.
            - descriptions: Descriptive summary of each cluster.
            - metrics: Clustering metrics like silhouette score.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(preprocessed_data)
    try:
        silhouette = silhouette_score(preprocessed_data, labels)
    except ValueError:
        silhouette = "Not applicable (only one cluster or noise)"
    insights = generate_cluster_insights(preprocessed_data, labels)
    descriptions = generate_cluster_descriptions(insights, scaler)
    return {
        "cluster_labels": labels,
        "cluster_insights": insights,
        "descriptions": descriptions,
        "metrics": {"silhouette_score": silhouette}
    }

def get_top_features_pca(data):
    """
    Identify the top features for clustering using PCA.

    Args:
        data (pd.DataFrame): Preprocessed data ready for clustering.

    Returns:
        pd.DataFrame.style: PCA loadings highlighting the most important features.
    """
    pca = PCA(n_components=2)
    pca.fit(data)
    loadings = pd.DataFrame(pca.components_, columns=data.columns, index=["PC1", "PC2"]).T
    return loadings.style.highlight_max(axis=0)

def generate_cluster_insights(data, labels):
    """
    Generate summary insights for each cluster.

    Args:
        data (pd.DataFrame): Preprocessed data ready for clustering.
        labels (array-like): Cluster labels for each data point.

    Returns:
        pd.DataFrame: Mean values for each feature per cluster.
    """
    data["Cluster"] = labels
    return data.groupby("Cluster").mean()

def generate_cluster_descriptions(insights, scaler):
    """
    Generate descriptive summaries for each cluster.

    Args:
        insights (pd.DataFrame): Cluster insights containing mean values per cluster.
        scaler (StandardScaler): Scaler used to standardize numeric data.

    Returns:
        pd.DataFrame: Descriptions for each cluster in human-readable format.
    """
    numeric_columns = scaler.feature_names_in_
    unscaled_data = scaler.inverse_transform(insights[numeric_columns])
    descriptions = [
        "; ".join([f"{col}: {val:.2f}" for col, val in zip(numeric_columns, row)])
        for row in unscaled_data
    ]
    return pd.DataFrame({"Cluster": insights.index, "Description": descriptions})

def display_results(results):
    """
    Display the clustering results, including metrics, insights, and descriptions.

    Args:
        results (dict): Dictionary containing clustering results and metrics.
    """
    st.success("Clustering completed!")

    # Metrics Section
    st.write("### Clustering Metrics")
    metrics = results["metrics"]
    st.write(f"**Silhouette Score:** {metrics['silhouette_score']}")

    insights_container = st.container()
    descriptions_container = st.container()

    # Insights Section
    with st.expander("View Cluster Insights"):
        st.dataframe(st.session_state.cluster_insights)

    # Descriptions Section
    with st.expander("View Cluster Descriptions"):
        st.dataframe(st.session_state.cluster_descriptions)

    # Download Buttons
    export_results_to_csv()

def export_results_to_csv():
    """
    Export clustering insights and descriptions as downloadable CSV files for the user.

    If available, the insights and descriptions are converted to CSV format and provided 
    as downloadable links via Streamlit.
    """
    if "cluster_insights" in st.session_state:
        insights_csv = st.session_state.cluster_insights.to_csv(index=False)
        st.download_button(
            label="Download Cluster Insights as CSV",
            data=insights_csv,
            file_name="cluster_insights.csv",
            mime="text/csv",
        )

    if "cluster_descriptions" in st.session_state:
        descriptions_csv = st.session_state.cluster_descriptions.to_csv(index=False)
        st.download_button(
            label="Download Cluster Descriptions as CSV",
            data=descriptions_csv,
            file_name="cluster_descriptions.csv",
            mime="text/csv",
        )

def handle_visualizations(preprocessed_data):
    """
    Handle the visualization options available for clustering results.

    Args:
        preprocessed_data (pd.DataFrame): The preprocessed dataset used for clustering.

    Displays different visualization options, including cluster distribution, PCA, and t-SNE.
    """
    if st.session_state.results is None:
        st.warning("Please run a clustering algorithm first.")
        return

    # Visualization options
    st.write("### Visualization Options")
    visualization_option = st.selectbox(
        "Choose a visualization method:",
        ["None", "Cluster Distribution", "PCA", "t-SNE"],
        key="selected_visualization"
    )

    if visualization_option == "Cluster Distribution":
        visualize_cluster_distribution_streamlit(st.session_state.results["cluster_labels"])
    elif visualization_option == "PCA":
        visualize_clusters_pca_streamlit(preprocessed_data, st.session_state.results["cluster_labels"])
    elif visualization_option == "t-SNE":
        visualize_clusters_tsne_streamlit(preprocessed_data, st.session_state.results["cluster_labels"], 30)

def visualize_cluster_distribution_streamlit(labels, ax=None):
    """
    Visualize the distribution of cluster sizes as a bar chart in Streamlit.

    Args:
        labels (array-like): Cluster labels assigned to each data point.
        ax (matplotlib.axes.Axes, optional): Axes to plot on (default is None).

    Displays the cluster size distribution and percentage of points in each cluster.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    percentages = [count / sum(counts) * 100 for count in counts]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(unique_labels, counts, color="skyblue")

    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Points")
    ax.set_xticks(unique_labels)

    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{percentage:.1f}%",
            ha="center",
            fontsize=10,
        )

    st.pyplot(fig)

def visualize_clusters_pca_streamlit(data, labels):
    """
    Visualize clusters in 2D using PCA and display feature contributions.

    Args:
        data (pd.DataFrame): Preprocessed data ready for clustering.
        labels (array-like): Cluster labels assigned to each data point.

    Displays a scatter plot of clusters in 2D space using PCA.
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    loadings = pd.DataFrame(pca.components_, columns=data.columns, index=["PC1", "PC2"])

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
    ax.set_title("2D Cluster Visualization (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    
    st.write("### Feature Contributions to Principal Components")
    st.dataframe(loadings.style.highlight_max(axis=1))
    st.pyplot(fig)

def visualize_clusters_tsne_streamlit(data, labels, perplexity=30, max_iter=300):
    """
    Visualize clustering results in 2D using t-SNE and display in Streamlit.

    Args:
        data (pd.DataFrame): Preprocessed data ready for clustering.
        labels (array-like): Cluster labels assigned to each data point.
        perplexity (int): Perplexity parameter for t-SNE (default is 30).
        max_iter (int): Maximum number of iterations for t-SNE optimization (default is 300).

    Displays a scatter plot of clusters in 2D space using t-SNE.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42)
    reduced_data = tsne.fit_transform(data)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)
    ax.set_title("2D Cluster Visualization (t-SNE)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    st.pyplot(fig)

if __name__ == "__main__":
    main()




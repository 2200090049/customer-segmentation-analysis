import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')


class FuturisticVisualizer:
    """
    Advanced visualization class with futuristic styling for customer segmentation analysis
    """

    def __init__(self, style='futuristic'):
        self.style = style
        self.setup_style()

    def setup_style(self):
        """Setup the visual styling theme"""
        if self.style == 'futuristic':
            # Futuristic color palette
            self.colors = {
                'primary': '#00D4FF',
                'secondary': '#FF6B6B',
                'accent': '#4ECDC4',
                'success': '#45B7D1',
                'warning': '#FFA726',
                'dark': '#2C3E50',
                'light': '#ECF0F1'
            }

            self.gradient_colors = [
                '#667eea', '#764ba2', '#f093fb', '#f5576c',
                '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
            ]

            # Set matplotlib style
            plt.style.use('dark_background')
            sns.set_palette("husl")

    def plot_dimensionality_reduction(self, data, features, labels, methods=['PCA', 'TSNE']):
        """Plot dimensionality reduction visualizations"""
        X = data[features].fillna(data[features].median())

        n_methods = len(methods)
        fig = make_subplots(
            rows=1, cols=n_methods,
            subplot_titles=[f'🔬 {method} Visualization' for method in methods]
        )

        for i, method in enumerate(methods):
            if method == 'PCA':
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                explained_var = reducer.explained_variance_ratio_
                title_suffix = f' (Var: {explained_var[0]:.2f}, {explained_var[1]:.2f})'
            elif method == 'TSNE':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                X_reduced = reducer.fit_transform(X)
                title_suffix = ''

            # Plot each cluster
            for cluster in np.unique(labels):
                if cluster == -1:
                    continue
                mask = labels == cluster
                fig.add_trace(
                    go.Scatter(
                        x=X_reduced[mask, 0], y=X_reduced[mask, 1],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(
                            size=8,
                            color=self.gradient_colors[cluster % len(self.gradient_colors)],
                            opacity=0.7
                        ),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=1, col=i + 1
                )

        fig.update_layout(
            height=500,
            title_text="🧬 Dimensionality Reduction Analysis",
            title_font_size=20,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig

    def plot_clustering_analysis(self, data, features, labels, model_name='K-Means'):
        """Create comprehensive clustering analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'🌌 3D {model_name} Clusters', f'💰 Income vs Spending ({model_name})',
                f'🎂 Age vs Spending ({model_name})', '📊 Cluster Size Distribution'
            ),
            specs=[[{"type": "scatter3d"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Create a copy of data with cluster labels
        plot_data = data.copy()
        plot_data['Cluster'] = labels

        # 3D Scatter plot
        if len(features) >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=plot_data[features[0]], y=plot_data[features[1]], z=plot_data[features[2]],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=labels,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Cluster")
                    ),
                    text=[f'Cluster {label}' for label in labels],
                    name='Customers'
                ),
                row=1, col=1
            )

        # Income vs Spending
        if 'annual_income' in features and 'spending_score' in features:
            for cluster in np.unique(labels):
                if cluster == -1:  # Skip noise points
                    continue
                cluster_data = plot_data[plot_data['Cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['annual_income'], y=cluster_data['spending_score'],
                        mode='markers', name=f'Cluster {cluster}',
                        marker=dict(size=8, color=self.gradient_colors[cluster % len(self.gradient_colors)])
                    ),
                    row=1, col=2
                )

        # Age vs Spending
        if 'age' in features and 'spending_score' in features:
            for cluster in np.unique(labels):
                if cluster == -1:
                    continue
                cluster_data = plot_data[plot_data['Cluster'] == cluster]
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['age'], y=cluster_data['spending_score'],
                        mode='markers', name=f'Cluster {cluster}',
                        marker=dict(size=8, color=self.gradient_colors[cluster % len(self.gradient_colors)]),
                        showlegend=False
                    ),
                    row=2, col=1
                )

        # Cluster size distribution
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_sizes.index if i != -1],
                y=[cluster_sizes[i] for i in cluster_sizes.index if i != -1],
                marker_color=self.gradient_colors[:len(cluster_sizes)],
                name='Cluster Size'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=900,
            title_text=f"🎯 {model_name} Clustering Analysis Results",
            title_font_size=24,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        return fig
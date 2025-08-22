import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class AdvancedCustomerSegmentation:
    """
    Advanced Customer Segmentation using multiple clustering algorithms
    with comprehensive evaluation metrics and visualization capabilities
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_params = None

    def prepare_data(self, data, features):
        """Prepare and scale data for clustering"""
        X = data[features].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X, X_scaled

    def find_optimal_clusters(self, X_scaled, max_clusters=10, methods=['elbow', 'silhouette']):
        """Find optimal number of clusters using multiple methods"""
        results = {}

        if 'elbow' in methods:
            results['elbow'] = self._elbow_method(X_scaled, max_clusters)

        if 'silhouette' in methods:
            results['silhouette'] = self._silhouette_method(X_scaled, max_clusters)

        if 'gap' in methods:
            results['gap'] = self._gap_statistic(X_scaled, max_clusters)

        return results

    def _elbow_method(self, X_scaled, max_clusters):
        """Elbow method for optimal cluster selection"""
        inertias = []
        k_range = range(1, max_clusters + 1)

        for k in k_range:
            if k == 1:
                inertias.append(np.sum(np.var(X_scaled, axis=0)))
            else:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)

        # Calculate elbow point using the "kneedle" method
        optimal_k = self._find_elbow_point(k_range, inertias)

        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'optimal_k': optimal_k
        }

    def _silhouette_method(self, X_scaled, max_clusters):
        """Silhouette analysis for optimal cluster selection"""
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        optimal_k = k_range[np.argmax(silhouette_scores)]

        return {
            'k_range': list(k_range),
            'scores': silhouette_scores,
            'optimal_k': optimal_k,
            'max_score': max(silhouette_scores)
        }

    def _find_elbow_point(self, k_range, inertias):
        """Find elbow point using the maximum curvature method"""
        if len(inertias) < 3:
            return k_range[0] if k_range else 2

        # Calculate the differences
        diffs = np.diff(inertias)
        diff2 = np.diff(diffs)

        # Find the point with maximum curvature (second derivative)
        if len(diff2) > 0:
            elbow_idx = np.argmax(np.abs(diff2)) + 2  # +2 because of double differencing
            return k_range[min(elbow_idx, len(k_range) - 1)]
        else:
            return k_range[1] if len(k_range) > 1 else k_range[0]

    def fit_multiple_models(self, X_scaled, n_clusters=5):
        """Fit multiple clustering models and compare performance"""
        models_to_fit = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10),
            'gmm': GaussianMixture(n_components=n_clusters, random_state=self.random_state),
            'hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }

        results = {}

        for name, model in models_to_fit.items():
            try:
                if name == 'gmm':
                    labels = model.fit_predict(X_scaled)
                else:
                    labels = model.fit_predict(X_scaled)

                # Store model and results
                self.models[name] = model

                # Calculate metrics (skip if only one cluster or all noise points)
                unique_labels = np.unique(labels)
                n_unique = len(unique_labels)

                if n_unique > 1 and not (n_unique == 1 and -1 in unique_labels):
                    metrics = self._calculate_metrics(X_scaled, labels)
                    results[name] = {
                        'model': model,
                        'labels': labels,
                        'n_clusters': n_unique,
                        'metrics': metrics
                    }
                else:
                    results[name] = {
                        'model': model,
                        'labels': labels,
                        'n_clusters': n_unique,
                        'metrics': {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
                    }

            except Exception as e:
                print(f"Error fitting {name}: {str(e)}")
                continue

        self.evaluation_results = results
        self._select_best_model()

        return results

    def _calculate_metrics(self, X_scaled, labels):
        """Calculate clustering evaluation metrics"""
        metrics = {}

        # Remove noise points for metric calculation (DBSCAN)
        mask = labels != -1
        if np.sum(mask) < 2:
            return {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}

        X_clean = X_scaled[mask]
        labels_clean = labels[mask]

        # Check if we have at least 2 clusters
        if len(np.unique(labels_clean)) < 2:
            return {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}

        try:
            metrics['silhouette'] = silhouette_score(X_clean, labels_clean)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
            metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
        except:
            metrics = {'silhouette': -1, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}

        return metrics

    def _select_best_model(self):
        """Select the best model based on evaluation metrics"""
        if not self.evaluation_results:
            return

        best_score = -float('inf')
        best_model_name = None

        for name, results in self.evaluation_results.items():
            # Composite score: prioritize silhouette score
            score = results['metrics']['silhouette']

            # Bonus for reasonable number of clusters
            n_clusters = results['n_clusters']
            if 2 <= n_clusters <= 8:
                score += 0.1

            if score > best_score:
                best_score = score
                best_model_name = name

        if best_model_name:
            self.best_model = best_model_name
            self.best_params = self.evaluation_results[best_model_name]

    def predict_cluster(self, X_new):
        """Predict cluster for new data points"""
        if not self.best_model or self.best_model not in self.models:
            raise ValueError("No trained model available")

        X_scaled = self.scaler.transform(X_new)
        model = self.models[self.best_model]

        if hasattr(model, 'predict'):
            return model.predict(X_scaled)
        else:
            # For models without predict method, find nearest cluster centers
            return self._predict_nearest_cluster(X_scaled)

    def _predict_nearest_cluster(self, X_scaled):
        """Predict cluster using nearest centroid approach"""
        if self.best_model == 'kmeans':
            return self.models[self.best_model].predict(X_scaled)
        else:
            # For other models, use a simple nearest neighbor approach
            # This is a simplified approach - in practice, you might want more sophisticated methods
            labels = self.best_params['labels']
            return np.zeros(len(X_scaled))  # Placeholder

    def get_cluster_centers(self):
        """Get cluster centers for the best model"""
        if not self.best_model:
            return None

        model = self.models[self.best_model]

        if hasattr(model, 'cluster_centers_'):
            return self.scaler.inverse_transform(model.cluster_centers_)
        elif hasattr(model, 'means_'):  # For GMM
            return self.scaler.inverse_transform(model.means_)
        else:
            return None

    def get_cluster_statistics(self, data, features, labels=None):
        """Get comprehensive statistics for each cluster"""
        if labels is None:
            if not self.best_model:
                raise ValueError("No trained model available")
            labels = self.best_params['labels']

        X = data[features].copy()
        cluster_stats = {}

        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue

            mask = labels == cluster_id
            cluster_data = X[mask]

            stats = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(data)) * 100,
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict(),
                'median': cluster_data.median().to_dict(),
                'min': cluster_data.min().to_dict(),
                'max': cluster_data.max().to_dict()
            }

            cluster_stats[f'Cluster_{cluster_id}'] = stats

        return cluster_stats

    def create_customer_personas(self, data, features, labels=None, additional_features=None):
        """Create detailed customer personas based on clusters"""
        if labels is None:
            if not self.best_model:
                raise ValueError("No trained model available")
            labels = self.best_params['labels']

        X = data[features].copy()
        if additional_features:
            for feature in additional_features:
                if feature in data.columns:
                    X[feature] = data[feature]

        personas = {}

        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue

            mask = labels == cluster_id
            cluster_data = X[mask]

            # Basic statistics
            persona = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(data)) * 100,
            }

            # Feature statistics
            for feature in features:
                persona[f'{feature}_mean'] = cluster_data[feature].mean()
                persona[f'{feature}_std'] = cluster_data[feature].std()
                persona[f'{feature}_median'] = cluster_data[feature].median()

            # Categorical features analysis
            if additional_features:
                for feature in additional_features:
                    if feature in cluster_data.columns and cluster_data[feature].dtype == 'object':
                        persona[f'{feature}_distribution'] = cluster_data[feature].value_counts().to_dict()

            # Generate persona description
            persona['description'] = self._generate_persona_description(persona, features)
            persona['business_value'] = self._assess_business_value(persona, features)
            persona['recommendations'] = self._generate_recommendations(persona, features)

            personas[f'Persona_{cluster_id}'] = persona

        return personas

    def _generate_persona_description(self, persona, features):
        """Generate human-readable persona description"""
        descriptions = []

        # Age-based description
        if 'age' in [f.split('_')[0] for f in persona.keys()]:
            avg_age = persona.get('age_mean', 0)
            if avg_age < 30:
                descriptions.append("Young adults")
            elif avg_age < 45:
                descriptions.append("Middle-aged adults")
            else:
                descriptions.append("Mature adults")

        # Income-based description
        if 'annual_income' in [f.split('_')[0] for f in persona.keys()]:
            avg_income = persona.get('annual_income_mean', 0)
            if avg_income < 40:
                descriptions.append("with lower income")
            elif avg_income < 70:
                descriptions.append("with moderate income")
            else:
                descriptions.append("with higher income")

        # Spending-based description
        if 'spending_score' in [f.split('_')[0] for f in persona.keys()]:
            avg_spending = persona.get('spending_score_mean', 0)
            if avg_spending < 35:
                descriptions.append("and conservative spending habits")
            elif avg_spending < 65:
                descriptions.append("and moderate spending patterns")
            else:
                descriptions.append("and high spending behavior")

        return " ".join(descriptions) if descriptions else "Customer segment"

    def _assess_business_value(self, persona, features):
        """Assess business value of the persona"""
        value_score = 0

        # Size matters
        if persona['percentage'] > 20:
            value_score += 2
        elif persona['percentage'] > 10:
            value_score += 1

        # High income is valuable
        if 'annual_income_mean' in persona:
            if persona['annual_income_mean'] > 70:
                value_score += 3
            elif persona['annual_income_mean'] > 50:
                value_score += 2

        # High spending is valuable
        if 'spending_score_mean' in persona:
            if persona['spending_score_mean'] > 70:
                value_score += 3
            elif persona['spending_score_mean'] > 50:
                value_score += 2

        if value_score >= 6:
            return "High Value"
        elif value_score >= 3:
            return "Medium Value"
        else:
            return "Low Value"

    def _generate_recommendations(self, persona, features):
        """Generate business recommendations for the persona"""
        recommendations = []

        # Income vs Spending analysis
        if 'annual_income_mean' in persona and 'spending_score_mean' in persona:
            income = persona['annual_income_mean']
            spending = persona['spending_score_mean']

            if income > 60 and spending < 40:
                recommendations.append("Target with premium products and exclusive offers")
                recommendations.append("Focus on value proposition and quality")
            elif income < 40 and spending > 60:
                recommendations.append("Offer affordable luxury and payment plans")
                recommendations.append("Emphasize deals and discounts")
            elif income > 60 and spending > 60:
                recommendations.append("Perfect target for premium products")
                recommendations.append("Implement loyalty programs")
            else:
                recommendations.append("Focus on value-for-money products")
                recommendations.append("Use promotional campaigns")

        # Size-based recommendations
        if persona['percentage'] > 25:
            recommendations.append("Major segment - prioritize in marketing campaigns")
        elif persona['percentage'] < 5:
            recommendations.append("Niche segment - consider specialized offerings")

        return recommendations

    def get_model_summary(self):
        """Get comprehensive model summary"""
        if not self.evaluation_results:
            return "No models have been trained yet"

        summary = {
            'models_evaluated': list(self.evaluation_results.keys()),
            'best_model': self.best_model,
            'best_model_metrics': self.best_params['metrics'] if self.best_params else {},
            'all_model_comparison': {}
        }

        for name, results in self.evaluation_results.items():
            summary['all_model_comparison'][name] = {
                'n_clusters': results['n_clusters'],
                'silhouette_score': results['metrics']['silhouette'],
                'calinski_harabasz_score': results['metrics']['calinski_harabasz'],
                'davies_bouldin_score': results['metrics']['davies_bouldin']
            }

        return summary
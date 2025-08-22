import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')


class CustomerDataProcessor:
    """
    A comprehensive data processor for customer segmentation analysis
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')

    def load_data(self, file_path):
        """Load customer data from CSV file"""
        try:
            data = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    def clean_data(self, data):
        """Clean and preprocess the raw data"""
        # Create a copy to avoid modifying original data
        df = data.copy()

        # Rename columns for consistency
        column_mapping = {
            'CustomerID': 'customer_id',
            'Genre': 'gender',
            'Gender': 'gender',
            'Age': 'age',
            'Annual Income (k$)': 'annual_income',
            'Spending Score (1-100)': 'spending_score'
        }

        # Apply column mapping if columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Handle missing values
        if df.isnull().sum().sum() > 0:
            print("⚠️ Missing values detected. Applying imputation...")
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])

        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_size:
            print(f"🔄 Removed {initial_size - len(df)} duplicate records")

        # Validate data ranges
        if 'age' in df.columns:
            df = df[(df['age'] >= 18) & (df['age'] <= 100)]

        if 'annual_income' in df.columns:
            df = df[df['annual_income'] > 0]

        if 'spending_score' in df.columns:
            df = df[(df['spending_score'] >= 1) & (df['spending_score'] <= 100)]

        print(f"✅ Data cleaning completed: {df.shape}")
        return df

    def create_features(self, data):
        """Create additional features for better segmentation"""
        df = data.copy()

        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                     bins=[0, 25, 35, 50, 65, 100],
                                     labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior', 'Elderly'])

        # Income categories
        if 'annual_income' in df.columns:
            df['income_category'] = pd.cut(df['annual_income'],
                                           bins=[0, 40, 70, 100, float('inf')],
                                           labels=['Low', 'Medium', 'High', 'Very_High'])

        # Spending categories
        if 'spending_score' in df.columns:
            df['spending_category'] = pd.cut(df['spending_score'],
                                             bins=[0, 35, 65, 100],
                                             labels=['Low_Spender', 'Medium_Spender', 'High_Spender'])

        # Income-to-Spending ratio
        if 'annual_income' in df.columns and 'spending_score' in df.columns:
            df['income_spending_ratio'] = df['annual_income'] / df['spending_score']
            df['spending_potential'] = df['annual_income'] * df['spending_score'] / 100

        print(f"✅ Feature engineering completed: {df.shape[1]} features")
        return df

    def prepare_clustering_features(self, data, features_for_clustering=None):
        """Prepare features specifically for clustering analysis"""
        if features_for_clustering is None:
            features_for_clustering = ['age', 'annual_income', 'spending_score']

        # Select features that exist in the data
        available_features = [col for col in features_for_clustering if col in data.columns]

        if not available_features:
            raise ValueError("No clustering features found in the data")

        clustering_data = data[available_features].copy()

        # Handle any remaining missing values
        clustering_data = clustering_data.fillna(clustering_data.median())

        # Standardize features
        clustering_data_scaled = self.scaler.fit_transform(clustering_data)

        print(f"✅ Clustering features prepared: {len(available_features)} features scaled")
        return clustering_data, clustering_data_scaled, available_features

    def get_data_summary(self, data):
        """Generate comprehensive data summary"""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': data.describe().to_dict() if len(
                data.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }

        # Categorical summary
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            summary['categorical_summary'][col] = data[col].value_counts().to_dict()

        return summary

    def detect_outliers(self, data, columns=None):
        """Detect outliers using IQR method"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        outliers = {}

        for col in columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers[col] = {
                    'count': outlier_mask.sum(),
                    'percentage': (outlier_mask.sum() / len(data)) * 100,
                    'indices': data[outlier_mask].index.tolist()
                }

        return outliers

    def correlation_analysis(self, data):
        """Perform correlation analysis on numeric features"""
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return None

        correlation_matrix = numeric_data.corr()

        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Threshold for high correlation
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        return {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations
        }

    def validate_data_quality(self, data):
        """Comprehensive data quality validation"""
        quality_report = {
            'total_records': len(data),
            'total_features': len(data.columns),
            'missing_data_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_records': data.duplicated().sum(),
            'data_types_consistent': True,
            'outliers': self.detect_outliers(data),
            'quality_score': 0
        }

        # Calculate quality score (0-100)
        score = 100

        # Deduct points for missing data
        if quality_report['missing_data_percentage'] > 0:
            score -= min(quality_report['missing_data_percentage'] * 2, 30)

        # Deduct points for duplicates
        duplicate_percentage = (quality_report['duplicate_records'] / len(data)) * 100
        score -= min(duplicate_percentage * 3, 20)

        # Deduct points for excessive outliers
        total_outliers = sum([outlier['count'] for outlier in quality_report['outliers'].values()])
        outlier_percentage = (total_outliers / len(data)) * 100
        if outlier_percentage > 10:
            score -= min((outlier_percentage - 10) * 2, 25)

        quality_report['quality_score'] = max(score, 0)

        return quality_report
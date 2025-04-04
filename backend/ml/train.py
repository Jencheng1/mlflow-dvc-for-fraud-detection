import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import dvc.api
from pathlib import Path

def load_data():
    """Load data using DVC."""
    try:
        # Load processed data using DVC
        processed_data_path = dvc.api.get_url('data/processed/transactions_processed.csv')
        df = pd.read_csv(processed_data_path)
        
        # Log data loading information
        mlflow.log_param("data_source", processed_data_path)
        mlflow.log_metric("data_size", len(df))
        mlflow.log_metric("fraud_rate", df['is_fraud'].mean())
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess data for training."""
    # Select features for training
    features = [
        'amount', 'hour', 'day_of_week',
        'merchant_category', 'location', 'device_type'
    ]
    
    X = df[features]
    y = df['is_fraud']
    
    # Log feature information
    mlflow.log_param("features", features)
    mlflow.log_param("target", "is_fraud")
    
    return X, y

def train_model():
    """Train model with versioning."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            # Load and preprocess data
            print("Loading data...")
            df = load_data()
            X, y = preprocess_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Log data split information
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Train model
            print("Training model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            })
            
            # Log feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            print("Model training completed. Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Save model locally
            model_dir = Path("data/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "fraud_detection_model.pkl"
            mlflow.sklearn.save_model(model, model_path)
            
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

if __name__ == "__main__":
    train_model() 
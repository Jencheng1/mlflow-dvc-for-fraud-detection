import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import dvc.api
import mlflow
from pathlib import Path

def generate_synthetic_data(n_samples=1000, fraud_rate=0.05, seed=42):
    """Generate synthetic transaction data with realistic patterns."""
    np.random.seed(seed)
    
    # Generate timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
    ) for _ in range(n_samples)]
    
    # Generate amounts with different distributions for fraud and legitimate transactions
    amounts = []
    for _ in range(n_samples):
        if np.random.random() < fraud_rate:
            # Fraudulent transactions tend to be larger
            amount = np.random.lognormal(mean=8, sigma=1)
        else:
            # Legitimate transactions follow a more normal distribution
            amount = np.random.lognormal(mean=6, sigma=0.5)
        amounts.append(amount)
    
    # Generate other features
    data = {
        'timestamp': timestamps,
        'amount': amounts,
        'merchant_category': np.random.choice(
            ['Retail', 'Food', 'Transport', 'Entertainment', 'Other', 'Online'],
            n_samples,
            p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        ),
        'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
        'location': np.random.choice(
            ['US', 'UK', 'EU', 'ASIA', 'LATAM'],
            n_samples,
            p=[0.4, 0.2, 0.2, 0.15, 0.05]
        ),
        'device_type': np.random.choice(
            ['Mobile', 'Desktop', 'Tablet'],
            n_samples,
            p=[0.6, 0.3, 0.1]
        ),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
    }
    
    return pd.DataFrame(data)

def save_data(df, data_dir='data'):
    """Save data with versioning using DVC."""
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    raw_path = os.path.join(data_dir, 'raw', 'transactions.csv')
    Path(os.path.dirname(raw_path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    
    # Save processed data
    processed_path = os.path.join(data_dir, 'processed', 'transactions_processed.csv')
    Path(os.path.dirname(processed_path)).mkdir(parents=True, exist_ok=True)
    
    # Add some basic processing
    processed_df = df.copy()
    processed_df['hour'] = processed_df['timestamp'].dt.hour
    processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
    processed_df['merchant_category'] = pd.Categorical(processed_df['merchant_category']).codes
    processed_df['location'] = pd.Categorical(processed_df['location']).codes
    processed_df['device_type'] = pd.Categorical(processed_df['device_type']).codes
    
    processed_df.to_csv(processed_path, index=False)
    
    return raw_path, processed_path

def log_data_versioning(raw_path, processed_path):
    """Log data versioning information to MLflow."""
    with mlflow.start_run(run_name=f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log data paths
        mlflow.log_param("raw_data_path", raw_path)
        mlflow.log_param("processed_data_path", processed_path)
        
        # Log data statistics
        df = pd.read_csv(raw_path)
        mlflow.log_metric("total_samples", len(df))
        mlflow.log_metric("fraud_rate", df['is_fraud'].mean())
        mlflow.log_metric("unique_customers", df['customer_id'].nunique())
        mlflow.log_metric("unique_merchants", df['merchant_category'].nunique())
        
        # Log data schema
        mlflow.log_dict(df.dtypes.to_dict(), "data_schema.json")

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Generate data
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=10000, fraud_rate=0.05)
    
    # Save data with versioning
    print("Saving data with versioning...")
    raw_path, processed_path = save_data(df)
    
    # Log data versioning information
    print("Logging data versioning information...")
    log_data_versioning(raw_path, processed_path)
    
    print("Data generation completed successfully!")
    print(f"Raw data saved to: {raw_path}")
    print(f"Processed data saved to: {processed_path}")

if __name__ == "__main__":
    main() 
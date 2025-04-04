import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import json

# Configure the page
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"

def main():
    st.title("🔍 Fraud Detection System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Transaction Analysis", "Model Performance", "About"]
    )
    
    if page == "Transaction Analysis":
        show_transaction_analysis()
    elif page == "Model Performance":
        show_model_performance()
    else:
        show_about()

def show_transaction_analysis():
    st.header("Transaction Analysis")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0)
        time = st.number_input("Time (24-hour format)", min_value=0.0, max_value=24.0)
    
    with col2:
        merchant_category = st.selectbox(
            "Merchant Category",
            ["Retail", "Food", "Transport", "Entertainment", "Other"]
        )
        location = st.text_input("Location")
    
    customer_id = st.text_input("Customer ID")
    
    if st.button("Analyze Transaction"):
        with st.spinner("Analyzing transaction..."):
            try:
                # Prepare transaction data
                transaction_data = {
                    "amount": amount,
                    "time": time,
                    "merchant_category": merchant_category,
                    "customer_id": customer_id,
                    "location": location
                }
                
                # Make API request
                response = requests.post(f"{API_URL}/predict", json=transaction_data)
                result = response.json()
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create three columns for metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        "Fraud Probability",
                        f"{result['fraud_probability']:.2%}",
                        delta=None
                    )
                
                with metric_col2:
                    st.metric(
                        "Prediction",
                        "Fraudulent" if result['is_fraud'] else "Legitimate",
                        delta=None
                    )
                
                with metric_col3:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']:.2%}",
                        delta=None
                    )
                
                # Add visualization
                fig = px.pie(
                    values=[result['fraud_probability'], 1 - result['fraud_probability']],
                    names=['Fraud Risk', 'Legitimate'],
                    title='Transaction Risk Assessment'
                )
                st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error analyzing transaction: {str(e)}")

def show_model_performance():
    st.header("Model Performance")
    
    try:
        # Fetch model information
        response = requests.get(f"{API_URL}/model-info")
        model_info = response.json()
        
        # Display model metrics
        st.subheader("Model Metrics")
        
        # Create four columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_info['metrics']['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{model_info['metrics']['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{model_info['metrics']['recall']:.2%}")
        with col4:
            st.metric("F1 Score", f"{model_info['metrics']['f1_score']:.2%}")
        
        # Display model version information
        st.subheader("Model Information")
        st.write(f"Version: {model_info['model_version']}")
        st.write(f"Last Updated: {model_info['last_updated']}")
        
    except Exception as e:
        st.error(f"Error fetching model information: {str(e)}")

def show_about():
    st.header("About")
    st.write("""
    This Fraud Detection System is built using:
    - Streamlit for the frontend interface
    - FastAPI for the backend API
    - MLflow for model versioning and tracking
    - DVC for data version control
    
    The system provides real-time fraud detection capabilities with:
    - Transaction analysis
    - Risk assessment
    - Model performance monitoring
    - Version control for both models and data
    """)

if __name__ == "__main__":
    main() 
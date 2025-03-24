import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go

# Configuration Before NGINX and Before Deployment
# API_ENDPOINT = "http://localhost:8000"
# TRANSACTION_ENDPOINT = f"{API_ENDPOINT}/transaction"
# PREDICTION_ENDPOINT = f"{API_ENDPOINT}/predict"

# Configuration After NGINX
# API_ENDPOINT = "http://localhost/api"
# TRANSACTION_ENDPOINT = f"{API_ENDPOINT}/transaction"
# PREDICTION_ENDPOINT = f"{API_ENDPOINT}/predict"

# Configuration after deployment
API_ENDPOINT = "http://34.202.237.111:8000"
TRANSACTION_ENDPOINT = f"{API_ENDPOINT}/transaction"
PREDICTION_ENDPOINT = f"{API_ENDPOINT}/predict"


# Set page config
st.set_page_config(
    page_title="Credit Card Transaction Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

def create_transaction(transaction_data):
    try:
        """Send transaction data to API and return response"""
        response = requests.post(
            TRANSACTION_ENDPOINT,
            json=transaction_data,
            headers={"Content-Type": "application/json"}
            timeout=10  # Add timeout
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return {"error": str(e)}

# def predict_fraud(transaction_id, event_timestamp=None):
def predict_fraud(transaction_id):
    """Request fraud prediction for a transaction"""
    request_data = {"transaction_id": transaction_id}
    # if event_timestamp:
    #     request_data["event_timestamp"] = event_timestamp
        
    response = requests.post(
        PREDICTION_ENDPOINT,
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    return response.json()

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:.2f}"

def display_transaction_form():
    """Display form for transaction input"""
    with st.form("transaction_form"):
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cc_num = st.text_input("Credit Card Number", value="4532640527811543")
            merchant = st.text_input("Merchant Name", value="Amazon")
            category = st.selectbox(
                "Category", 
                ["shopping", "food_dining", "entertainment", "travel", "grocery", "health_fitness", "misc_pos"]
            )
            amt = st.number_input("Amount", min_value=1.0, max_value=10000.0, value=125.43)
            merch_lat = st.number_input("Merchant Latitude", value=40.7580)
            merch_long = st.number_input("Merchant Longitude", value=-73.9855)
        
        with col2:
            first = st.text_input("First Name", value="John")
            last = st.text_input("Last Name", value="Doe")
            gender = st.selectbox("Gender", ["M", "F"])
            dob = st.date_input("Date of Birth", value=datetime.date(1985, 6, 15))
            job = st.text_input("Job", value="Engineer")
            
        col3, col4 = st.columns(2)
        
        with col3:
            street = st.text_input("Street Address", value="123 Main St")
            city = st.text_input("City", value="New York")
            state = st.text_input("State", value="NY")
            zip_code = st.number_input("ZIP Code", value=10001, min_value=10000, max_value=99999)
            
        with col4:
            lat = st.number_input("Customer Latitude", value=40.7128)
            long = st.number_input("Customer Longitude", value=-74.0060)
            city_pop = st.number_input("City Population", value=8336817)
        
        submitted = st.form_submit_button("Submit Transaction")
        
        if submitted:
            # Create transaction data dictionary
            transaction_data = {
                "cc_num": int(cc_num),
                "merchant": merchant,
                "category": category,
                "amt": float(amt),
                "first": first,
                "last": last,
                "gender": gender,
                "street": street,
                "city": city,
                "state": state,
                "zip": int(zip_code),
                "lat": float(lat),
                "long": float(long),
                "city_pop": int(city_pop),
                "job": job,
                "dob": dob.strftime("%Y-%m-%d"),
                "merch_lat": float(merch_lat),
                "merch_long": float(merch_long)
            }
            
            return transaction_data
    
    return None

def display_results(transaction_response, prediction_response):
    """Display transaction and prediction results"""
    # Create 3 columns for displaying results
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Transaction Details")
        st.write(f"Transaction ID: {transaction_response['transaction_id']}")
        st.write(f"MongoDB ID: {transaction_response['mongodb_id']}")
        st.write(f"Status: {transaction_response['status']}")
    
    with col2:
        st.subheader("Fraud Analysis")
        # fraud_probability = prediction_response["fraud_probability"] * 100
        # st.write(f"Fraud probability: {prediction_response["fraud_probability"] * 100}")
        fraud_label = prediction_response["fraud_label"]
        st.write(f"Fraud label: {prediction_response['fraud_label']}")
        
        # Display fraud probability with gauge chart
        # fig = go.Figure(go.Indicator(
        #     mode="gauge+number",
        #     value=fraud_probability,
        #     title={"text": "Fraud Probability"},
        #     domain={"x": [0, 1], "y": [0, 1]},
        #     gauge={
        #         "axis": {"range": [0, 100]},
        #         "bar": {"color": "darkblue"},
        #         "steps": [
        #             {"range": [0, 33], "color": "green"},
        #             {"range": [33, 66], "color": "yellow"},
        #             {"range": [66, 100], "color": "red"},
        #         ],
        #         "threshold": {
        #             "line": {"color": "red", "width": 4},
        #             "thickness": 0.75,
        #             "value": 50,
        #         },
        #     },
        # ))
        # st.plotly_chart(fig)
        
        # Display fraud label
        if fraud_label:
            st.error("⚠️ FRAUD DETECTED")
        else:
            st.success("✅ TRANSACTION APPEARS LEGITIMATE")
    
    # with col3:
    #     st.subheader("Risk Factors")
        
    #     # Display some common risk factors
    #     risk_factors = {
    #         "Transaction Amount": prediction_response["fraud_probability"] > 0.7 and "High" or "Normal",
    #         "Customer Location": prediction_response["fraud_probability"] > 0.5 and "Suspicious" or "Normal",
    #         "Merchant Category": prediction_response["fraud_probability"] > 0.6 and "High Risk" or "Low Risk",
    #         "Time of Transaction": prediction_response["fraud_probability"] > 0.4 and "Unusual" or "Normal"
    #     }
        
    #     for factor, status in risk_factors.items():
    #         if "High" in status or "Suspicious" in status or "Unusual" in status:
    #             st.warning(f"{factor}: {status}")
    #         else:
    #             st.info(f"{factor}: {status}")

def display_transaction_history():
    """Display fake transaction history for visualization"""
    st.subheader("Recent Transaction History")
    
    # Create sample transaction data
    data = {
        "Date": pd.date_range(start="2025-03-10", periods=7, freq="D"),
        "Amount": [120.50, 34.95, 550.00, 10.75, 212.46, 125.43, 72.15],
        "Merchant": ["Target", "Starbucks", "Apple", "7-Eleven", "Best Buy", "Amazon", "Uber"],
        "Category": ["shopping", "food_dining", "electronics", "grocery", "electronics", "shopping", "travel"],
        # "Fraud_Probability": [0.02, 0.01, 0.15, 0.03, 0.08, 0.05, 0.02]
    }
    
    history_df = pd.DataFrame(data)
    # history_df["Fraud_Probability"] = history_df["Fraud_Probability"] * 100
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of transaction amounts by category
        fig1 = px.bar(
            history_df, 
            x="Category", 
            y="Amount",
            color="Amount",
            title="Transaction Amounts by Category",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig1)
    
    # with col2:
    #     # Scatter plot of fraud probability vs amount
    #     fig2 = px.scatter(
    #         history_df,
    #         x="Amount",
    #         y="Fraud_Probability",
    #         color="Category",
    #         size="Fraud_Probability",
    #         hover_data=["Merchant", "Date"],
    #         title="Fraud Risk vs Transaction Amount"
    #     )
    #     fig2.add_shape(
    #         type="line",
    #         x0=0, y0=50,
    #         x1=600, y1=50,
    #         line=dict(color="Red", width=2, dash="dash"),
    #     )
    #     st.plotly_chart(fig2)
    
    # Show the data table
    st.write("Transaction Details")
    st.dataframe(
        history_df.style.format({
            "Amount": "${:.2f}",
            "Fraud_Probability": "{:.2f}%"
        })
    )

def main():
    """Main application function"""
    # App title and introduction
    st.title("Credit Card Transaction Fraud Detection Machine Learning System")
    
    # Simple banner image instead of Lottie animation
    st.image("https://img.freepik.com/free-vector/hand-drawn-flat-design-ransomware-illustration_23-2149373424.jpg?t=st=1742129010~exp=1742132610~hmac=1173f73e5f82078cf082765b7660b18bf0133981f47eec7840263cf65841dc12&w=1380", width=300)
    
    st.markdown("""
    This application demonstrates real-time credit card fraud detection using machine learning.
    Enter transaction details below to check if a transaction might be fraudulent.
    """)
    
    # Create tabs for different functions
    tab1, tab2 = st.tabs(["Process Transaction", "Transaction History"])
    
    with tab1:
        transaction_data = display_transaction_form()
        
        if transaction_data:
            with st.spinner("Processing transaction..."):
                # Submit transaction
                try:
                    transaction_response = create_transaction(transaction_data)
                    st.write("Transaction Response:", transaction_response)  # Debugging output

                    
                    # Short delay for effect
                    time.sleep(1)
                    
                    # Get prediction

                    # Extracting transaction_id 
                    transaction_id = transaction_response["transaction_id"]
                    if transaction_id:
                        prediction_response = predict_fraud(transaction_id)
                        st.write("Prediction Response:", prediction_response)  # Debugging output
                    else:
                        st.error("Transaction ID is None. The transaction was not created properly.")
                    # prediction_response = predict_fraud(transaction_id)
                    
                    # Display results
                    display_results(transaction_response, prediction_response)
                except Exception as e:
                    st.error(f"Error processing transaction: {str(e)}")
                    st.info("Make sure the FastAPI backend is running at " + API_ENDPOINT)
    
    with tab2:
        display_transaction_history()
    
    # Display app info in sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This application connects to a FastAPI backend that processes 
        credit card transactions and predicts potential fraud using machine learning.
        
        The system uses:
        - Feast for feature retrieval
        - MLflow for model management
        - MongoDB for transaction storage
        """)
        
        # st.subheader("System Status")
        # col1, col2 = st.columns(2)
        # col1.metric("API Status", "Online", "100%")
        # col2.metric("Model Version", "v5", "Latest")
        
        # Display some fake stats
        st.subheader("System Statistics")
        st.write("Transactions Processed: 12,453")
        st.write("Fraud Detected: 187")
        st.write("Accuracy Rate: 99.2%")
        
        # Add instructions
        st.subheader("Instructions")
        st.markdown("""
        1. Enter transaction details in the form
        2. Click "Submit Transaction" to process
        3. View fraud analysis results
        4. Check transaction history tab for trends
        
        """)

if __name__ == "__main__":
    main()
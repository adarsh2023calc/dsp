

import requests
import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random
import string
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,roc_curve, auc,ConfusionMatrixDisplay,confusion_matrix
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
def extract_data(url):
    # Initialize parameters for the API request
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,  # Maximum items per page
        "page": 1,        # Start with page 1
        "sparkline": False
    }

    crypto_list = []  # List to store cryptocurrency data

    try:
        # Loop to fetch 1000 records (4 pages of 250 each)
        for page in range(1, 7):  # Pages 1 to 6
            params["page"] = page
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            data = response.json()

            # Extract required fields
            for coin in data:
                crypto_list.append({
                    "Name": coin["name"],
                    "Symbol": coin["symbol"].upper(),
                    "Price (USD)": coin["current_price"],
                    "Market Cap": coin["market_cap"],
                    "Trading Volume (24h)": coin["total_volume"],
                    "Price Change (24h %)": coin["price_change_percentage_24h"],
                    "ATH (All-Time High)": coin["ath"],
                    "ATH Change (%)": coin["ath_change_percentage"],
                    "Low (24h)": coin["low_24h"],
                    "High (24h)": coin["high_24h"],
                    "Market Cap Rank": coin["market_cap_rank"],
                    "Total Supply": coin.get("total_supply", "N/A"),
                    "Circulating Supply": coin.get("circulating_supply", "N/A"),
                    "Max Supply": coin.get("max_supply", "N/A"),
                    "Last Updated": coin["last_updated"],
                    "Whether to invest": 0 if np.random.uniform(0, 1) < 0.5 else 1,
                    "Category": np.random.choice(["DeFi", "NFT", "Payment", "Gaming", "Others"]),
                    "Region": np.random.choice(["Global", "Asia", "USA", "Europe"]),
                    "Risk Level": np.random.choice(["High", "Medium", "Low"]),
                    "Launch Year": np.random.choice(["Before 2015", "2015-2020", "After 2020"])
                })

        # Convert to a Pandas DataFrame for better visualization
        df = pd.DataFrame(crypto_list)
        print(f"Successfully fetched {len(df)} records.")
        print(df.head())  # Display the first few records

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


    # Generate synthetic data
    count = 2000  # Number of synthetic records to generate
    for _ in range(count):
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        symbol = ''.join(random.choices(string.ascii_uppercase, k=3))
        crypto_list.append({
            "Name": f"FakeCoin-{name}",
            "Symbol": symbol,
            "Price (USD)": round(random.uniform(0.01, 1000), 2),
            "Market Cap": random.randint(10000, 1000000000),
            "Trading Volume (24h)": random.randint(1000, 1000000),
            "Price Change (24h %)": round(random.uniform(-50, 50), 2),
            "ATH (All-Time High)": round(random.uniform(1, 2000), 2),
            "ATH Change (%)": round(random.uniform(-100, 0), 2),
            "Low (24h)": round(random.uniform(0.01, 900), 2),
            "High (24h)": round(random.uniform(1, 1000), 2),
            "Market Cap Rank": random.randint(1001, 2000),
            "Total Supply": random.randint(10000, 100000),
            "Circulating Supply": random.randint(1000, 100000),
            "Max Supply": random.randint(1000, 10000000),
            "Last Updated": (datetime.utcnow() - timedelta(seconds=random.randint(0, 86400))).isoformat(),
            "Whether to invest": 0 if np.random.uniform(0, 1) < 0.5 else 1,
            "Category": np.random.choice(["DeFi", "NFT", "Payment", "Gaming", "Others"]),
            "Region": np.random.choice(["Global", "Asia", "USA", "Europe"]),
            "Risk Level": np.random.choice(["High", "Medium", "Low"]),
            "Launch Year": np.random.choice(["Before 2015", "2015-2020", "After 2020"])
        })

    # Convert to a Pandas DataFrame for better visualization
    df = pd.DataFrame(crypto_list)
    print(f"Successfully fetched {len(df)} records (including {count} synthetic records).")
    return df



# Function definitions remain largely the same with the improvements outlined

def display_top_5cryptos(df):
    df = df.sort_values(by='Trading Volume (24h)', ascending=False)
    st.subheader("📊 Top 5 Trading Cryptocurrencies")
    df1 = df.head(5)
    
    # Display data
    st.dataframe(df1)
    df = df.sort_values(by='Price (USD)', ascending=False)
    df1 = df.head(5)
    # Create an interactive bar chart using Plotly
    fig = px.bar(
        df1,
        x="Name",
        y="Trading Volume (24h)",
        color="Risk Level",
        title="Top 5 Cryptocurrencies by Trading Volume",
        labels={"Trading Volume (24h)": "Trading Volume (USD)"}
    )
    st.plotly_chart(fig)
    fig = px.bar(
        df1,
        x="Name",
        y="Price (USD)",
        color="Whether to invest",
        title="Top 5 Cryptocurrencies by Price based on whether to invest or not",
        labels={"Trading Price": "(USD)"}
    )

    st.plotly_chart(fig)
def display_cols_having_null_values(df):
    st.subheader("🛠 Columns with Null Values")
    null_values = df.isnull().sum()
    st.write(null_values)
    null_cols = null_values[null_values > 0]
    if null_cols.empty:
        st.success("No columns with null values!")
    else:
        st.write(null_cols)

def display_any_outliers(df):
    st.subheader("📉 Outlier Detection")
    cols = [
        'Price (USD)', 'Market Cap', 'Trading Volume (24h)',
        'Price Change (24h %)', 'ATH (All-Time High)', 'ATH Change (%)',
        'Low (24h)', 'High (24h)', 'Market Cap Rank',
        'Total Supply', 'Circulating Supply', 'Max Supply'
    ]
    selected_feature = st.selectbox("Select a feature to visualize:", cols)
    
    # Plotting
    fig = px.box(df, y=selected_feature, title=f"Boxplot of {selected_feature}")
    st.plotly_chart(fig)

def display_heatmap(df):
    st.subheader("🔍 Correlation Heatmap")
    cols = [
        'Price (USD)', 'Market Cap', 'Trading Volume (24h)', 
        'Price Change (24h %)', 'ATH (All-Time High)', 'ATH Change (%)', 
        'Low (24h)', 'High (24h)', 'Market Cap Rank', 
        'Total Supply', 'Circulating Supply', 'Max Supply'
    ]
    
    
    corr = df[cols].corr()

    # Display header and description
    st.header("Correlation between the features")
    st.text("We use the sns library to display the heatmap")

    # Create a matplotlib figure for the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    
    # Display the heatmap in Streamlit
    st.pyplot(fig) 
    
    st.plotly_chart(fig)

def apply_pca(df):
    df=df.dropna()
    st.subheader("🎨 PCA Visualization")
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns
    pca = PCA(n_components=3)
    result = pca.fit_transform(df[num_columns])
    df['PC 1'], df['PC 2'], df['PC 3'] = result[:, 0], result[:, 1], result[:, 2]
    total_var = pca.explained_variance_ratio_.sum() * 100
    
    fig = px.scatter_3d(
        df, x='PC 1', y='PC 2', z='PC 3', color='Risk Level',
        title=f'PCA: Total Explained Variance {total_var:.2f}%',
        labels={"PC 1": "Principal Component 1", "PC 2": "Principal Component 2", "PC 3": "Principal Component 3"}
    )
    st.plotly_chart(fig)

def preprcoess_data(df):
    st.subheader("⚙️ Data Preprocessing")
    st.write(f"Before Preprocessing: {df.shape[0]} rows")
    df = df.dropna()
    df = df.drop_duplicates()
    st.write(f"After Preprocessing: {df.shape[0]} rows")
    return df

def convert_categorical(df):
        
    cols=['Risk Level']

    enc=LabelEncoder()
    df['Risk Level']=enc.fit_transform(df['Risk Level'])

    cols=['Category','Region']

    extra = pd.get_dummies(df[cols])

    df[extra.columns.values]=extra
    df=df.drop(columns=cols)

    return df


def display_predictions(df):
    st.title("Crypto Investment Prediction")
    st.header("Select a Model and Analyze Predictions")
    
    # Preprocess the dataset
    df = convert_categorical(df)
    df= preprcoess_data(df)
    y = df['Whether to invest']  # Target variable
    cols = ['Name', 'Symbol', 'Launch Year', 'Whether to invest', 'Market Cap', 'Last Updated']
    X = df.drop(columns=cols)

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model selection
    models = ['Logistic Regression', 'Support Vector Machine', 'Random Forest Classification','LGBMClassifier']
    model_map = {
        'Logistic Regression': LogisticRegressionCV(random_state=42),
        'Support Vector Machine': SVC(C=0.1, kernel='rbf', probability=True),
        'Random Forest Classification': RandomForestClassifier(n_estimators=100, random_state=42),
        "LGBMClassifier":LGBMClassifier()
    }
    selected_model = st.selectbox("Choose a model for prediction:", models)

    # Train the selected model
    model = model_map[selected_model]
    model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    y_pred = model.predict(X_test)

    st.subheader(f"Results for {selected_model}")
    st.write(f"**Train Accuracy:** {train_accuracy:.2f}")
    st.write(f"**Test Accuracy:** {test_accuracy:.2f}")

    # Confusion Matrix
    st.header("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Yes", "No"])
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
    st.pyplot(fig_cm)

    # ROC-AUC Curve
    st.header("ROC-AUC Curve")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)





if __name__ == "__main__":
    st.title("🚀 Cryptocurrency Dashboard")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose a section:", [
        "Introduction", "Top 5 Cryptos", "Null Value Analysis", 
        "Outlier Detection", "Data Preprocessing", 
        "PCA Visualization", "Correlation Heatmap","Machine Learning Model"
    ])
    
    # Load Data
    url = "https://api.coingecko.com/api/v3/coins/markets"
    df = extract_data(url)
    
    if options == "Introduction":
        st.header("Introduction")
        st.markdown("""
        This app showcases cryptocurrency analysis using the CoinGecko API and synthetic data. 
        Key features include identifying top-performing cryptos, handling missing data, 
        and advanced visualizations like PCA and heatmaps.
        """)
    elif options == "Top 5 Cryptos":
        display_top_5cryptos(df)
    elif options == "Null Value Analysis":
        display_cols_having_null_values(df)
    elif options == "Outlier Detection":
        display_any_outliers(df)
    elif options == "Data Preprocessing":
        df = preprcoess_data(df)
    elif options == "PCA Visualization":
        apply_pca(df)
    elif options == "Correlation Heatmap":
        display_heatmap(df)
    
    elif options =="Machine Learning Model":
        display_predictions(df)

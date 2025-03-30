import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Streamlit Page Config (Must be first command)
st.set_page_config(page_title="DDoS Attack Detector", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")  # Ensure dataset.csv is in the same directory
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Ensure correct columns are selected
available_columns = df.columns.tolist()
important_columns = [
    "pktcount", "bytecount", "dur", "tot_dur", "flows", 
    "packetins", "pktperflow", "byteperflow", "pktrate", "tot_kbps"
]
important_columns = [col for col in important_columns if col in available_columns]

if "label" in available_columns:
    df = df[important_columns + ["label"]]
else:
    df = df[important_columns]

# Convert categorical features to numeric
label_encoders = {}
for col in df.select_dtypes(exclude=[np.number]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Preprocessing
X = df.drop(columns=["label"]) if "label" in df.columns else df  # Features
y = df["label"] if "label" in df.columns else None  # Target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) if not X.empty else X
if y is not None and not X.empty:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = None, None, None, None

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
if y is not None and X_train is not None:
    model.fit(X_train, y_train)

# Custom CSS
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; }
        h1 { color: #1f77b4; text-align: center; }
        .stButton button { background-color: #1f77b4; color: white; padding: 10px 20px; }
        .stButton button:hover { background-color: #155a8a; }
    </style>
    """, unsafe_allow_html=True
)

# Title
st.title("🌐 DDoS Attack Detection System")

st.write("🚀 Enter network traffic features to predict if it's a **normal request** or a **DDoS attack**.")

# Sidebar Instructions
st.sidebar.header("📌 Instructions")
st.sidebar.write("1️⃣ Enter network features.")
st.sidebar.write("2️⃣ Click '**Predict**' to classify traffic.")
st.sidebar.write("3️⃣ View the result below.")

# Mapping for meaningful labels
feature_labels = {
    "pktcount": "Packet Count",
    "bytecount": "Byte Count",
    "dur": "Duration (ms)",
    "tot_dur": "Total Duration (ms)",
    "flows": "Number of Flows",
    "packetins": "Packet-Ins",
    "pktperflow": "Packets Per Flow",
    "byteperflow": "Bytes Per Flow",
    "pktrate": "Packet Rate (packets/sec)",
    "tot_kbps": "Total Bandwidth (kbps)"
}

# Input Fields
st.header("📊 Enter Traffic Features")
user_inputs = {}
columns = important_columns
cols = st.columns(2)

for i, col_name in enumerate(columns):
    with cols[i % 2]:  # Distribute inputs across 2 columns
        user_inputs[col_name] = st.number_input(f"{feature_labels.get(col_name, col_name)}", value=0.0)

# Prediction Function
def predict_attack(features):
    try:
        data = np.array([features[col] for col in columns]).reshape(1, -1)
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Predict Button
if st.button("🔍 Predict"):
    prediction = predict_attack(user_inputs)
    if prediction is not None:
        st.subheader("🛑 Prediction Result")
        if prediction == 1:
            st.error("🚨 **DDoS Attack Detected!** 🚨")
        else:
            st.success("✅ **Normal Traffic**")
        
        # Pie Chart
        labels = ["DDoS Attack", "Normal Traffic"]
        sizes = [prediction, 1 - prediction]
        colors = ["#ff6347", "#66cdaa"]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)
    else:
        st.warning("Unable to generate a prediction. Please check your inputs and try again.")
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-msg {
        padding: 20px;
        background-color: #d4edda;
        color: #155724;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .error-msg {
        padding: 20px;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    # Standard features: V1-V28 + Amount
    st.session_state.feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-cards.png", width=80)
st.sidebar.title("Configuration")

app_mode = st.sidebar.selectbox("Choose Section", 
    ["Home", "Data Upload", "Exploratory Data Analysis", "Modeling", "Prediction", "Conclusion"])

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
if st.sidebar.button("Reload Session"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 1. HOME / DASHBOARD
# ==========================================
if app_mode == "Home":
    st.markdown('<p class="main-header">💳 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Detect fraud in card transactions.")
        st.write("""
        This system utilizes **Machine Learning** to classify transactions as **Normal** or **Fraudulent**.
        
        **Key Features:**
        - 📊 Interactive EDA 
        - 🤖 Model Training 
        - ⚡ Real-time Prediction
        """)
        st.info("👈 Start by uploading your dataset in the **Data Upload** section.")

    with col2:
        st.image("ECU-Logo.png", 
                 caption="Secure Transactions", use_container_width=True)

    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("📊 Current Data Snapshot")
        df = st.session_state.df
        row_count = len(df)
        fraud_count = len(df[df['Class'] == 1])
        fraud_percentage = (fraud_count / row_count) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", f"{row_count:,}")
        c2.metric("Fraud Cases", f"{fraud_count:,}")
        c3.metric("Fraud Percentage", f"{fraud_percentage:.4f}%")

# ==========================================
# 2. DATA UPLOAD SECTION
# ==========================================
elif app_mode == "Data Upload":
    st.title("📂 Data Upload")
    st.markdown("Upload your `creditcard.csv` file.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            try:
                # Read CSV using Pandas
                df = pd.read_csv(uploaded_file)
                
                if 'Class' not in df.columns:
                    st.error("Dataset must contain a 'Class' column (0 for Normal, 1 for Fraud).")
                else:
                    st.session_state.df = df
                    st.success("Data loaded successfully!")
                    
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif app_mode == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    
    if st.session_state.df is None:
        st.warning("Please upload data first.")
    else:
        df = st.session_state.df
        
        tabs = st.tabs(["Statistics", "Variable Distribution", "Relationships", "Missing Values"])
        
        with tabs[0]:
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe())
            
            st.subheader("Class Balance")
            class_counts = df['Class'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=class_counts.index, y=class_counts.values, palette=['#1f77b4', '#d62728'], ax=ax)
            ax.set_title("Transaction Class Distribution (0: Normal, 1: Fraud)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
        with tabs[1]:
            st.subheader("Distributions")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Transaction Amount Distribution**")
                fig1, ax1 = plt.subplots()
                sns.histplot(df['Amount'], bins=50, kde=True, color='blue', ax=ax1)
                ax1.set_xlim(0, 2000)
                st.pyplot(fig1)
            with col2:
                st.write("**Time Distribution**")
                fig2, ax2 = plt.subplots()
                sns.histplot(df['Time'], bins=50, kde=True, color='green', ax=ax2)
                st.pyplot(fig2)
                
            st.write("**V-Features Distribution**")
            # Limit list to first 10 for UI cleanliness
            selected_feat = st.selectbox("Select Feature to Visualize", st.session_state.feature_columns[:10])
            fig3, ax3 = plt.subplots()
            # Sample data for plotting speed if dataset is huge
            plot_df = df.sample(min(10000, len(df)))
            sns.histplot(data=plot_df, x=selected_feat, hue="Class", kde=True, palette={0:'blue', 1:'red'}, element="step", ax=ax3)
            st.pyplot(fig3)

        with tabs[2]:
            st.subheader("Correlation Heatmap")
            with st.spinner("Generating heatmap..."):
                fig, ax = plt.subplots(figsize=(10, 8))
                # Subsample for heatmap speed
                corr_df = df.sample(min(5000, len(df)))
                corr = corr_df.corr()
                sns.heatmap(corr, cmap='coolwarm_r', annot=False, ax=ax)
                st.pyplot(fig)

        with tabs[3]:
            st.subheader("Missing Values Check")
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                st.write(missing_counts[missing_counts > 0])
            else:
                st.success("No missing values found in the dataset! ✅")

# ==========================================
# 4. MODELING SECTION
# ==========================================
elif app_mode == "Modeling":
    st.title("⚙️ Model Training & Evaluation")
    
    if st.session_state.df is None:
        st.warning("Please upload data first.")
    else:
        st.sidebar.subheader("Model Configuration")
        model_type = st.sidebar.selectbox("Select Algorithm", 
            ["Decision Tree", "Random Forest", "Logistic Regression"])
        
        split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7)
        
        # Hyperparameters
        params = {}
        if model_type == "Random Forest":
            params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 100, 20)
            params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
        elif model_type == "Decision Tree":
            params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
        elif model_type == "Logistic Regression":
            params['max_iter'] = st.sidebar.slider("Max Iterations", 100, 500, 100)

        if st.button("🚀 Train Model"):
            with st.spinner(f"Training {model_type}..."):
                
                df = st.session_state.df
                X = df[st.session_state.feature_columns]
                y = df['Class']
                
                # Split Data
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=42, stratify=y)
                
                # Select Model
                if model_type == "Decision Tree":
                    clf = DecisionTreeClassifier(max_depth=params['max_depth'], random_state=42)
                elif model_type == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)
                else:
                    # Logistic Regression usually benefits from scaling
                    clf = Pipeline([
                        ('scaler', StandardScaler()),
                        ('logreg', LogisticRegression(max_iter=params['max_iter'], random_state=42))
                    ])
                
                # Train
                clf.fit(X_train, y_train)
                st.session_state.model = clf
                
                # Predict
                y_pred = clf.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                st.success(f"{model_type} Trained Successfully!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Precision", f"{prec:.4f}")
                col3.metric("Recall", f"{rec:.4f}")
                col4.metric("F1-Score", f"{f1:.4f}")
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
                st.pyplot(fig)

# ==========================================
# 5. PREDICTION SECTION
# ==========================================
elif app_mode == "Prediction":
    st.title("⚡ Real-time Prediction")
    
    if st.session_state.model is None:
        st.error("No model trained yet! Please go to the 'Modeling' tab and train a model.")
    else:
        st.markdown("### Test the model with transaction data")
        
        input_method = st.radio("Choose Input Method", ["Select Random Test Transaction", "Manual Input (Simulation)"])
        
        input_data = None
        
        if input_method == "Select Random Test Transaction":
            if st.session_state.df is not None and st.button("🎲 Pick Random Transaction"):
                # Pick random sample
                sample = st.session_state.df.sample(1)
                st.write("Selected Transaction Data:")
                st.dataframe(sample)
                
                # Extract features for prediction
                input_data = sample[st.session_state.feature_columns]
                actual_label = sample['Class'].values[0]
                st.caption(f"Actual Label: {'Fraud' if actual_label == 1 else 'Normal'}")

        else:
            st.info("Inputting 30 features manually is tedious. We will simulate a simplified input.")
            col1, col2 = st.columns(2)
            amount = col1.number_input("Amount", value=0.0)
            v1 = col2.number_input("V1 (Principal Component)", value=0.0)
            
            if st.button("Construct Transaction"):
                # Create a dummy dataframe with 0s
                data = {col: 0.0 for col in st.session_state.feature_columns}
                data['Amount'] = amount
                data['V1'] = v1
                input_data = pd.DataFrame([data])
                st.write("Simulated Input (other V-features set to 0):")
                st.dataframe(input_data)
        
        if input_data is not None:
            if st.button("Check Transaction Result"):
                model = st.session_state.model
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                if prediction == 1:
                    st.markdown(f'<div class="error-msg">🟥 Warning: FRAUD Detected!</div>', unsafe_allow_html=True)
                    st.write(f"**Probability of Fraud:** {probability[1]:.4f}")
                else:
                    st.markdown(f'<div class="success-msg">🟩 Normal Transaction</div>', unsafe_allow_html=True)
                    st.write(f"**Confidence:** {probability[0]:.4f}")
                    st.balloons()

# ==========================================
# 6. CONCLUSION
# ==========================================
elif app_mode == "Conclusion":
    st.title("📝 Conclusion")
    # st.markdown("""
    # ### Project Summary
    # We implemented a Fraud Detection System.
    
    # **Insights:**
    # - **Imbalanced Data:** Fraud cases are very rare (<1%).
    # - **Model Selection:** Random Forest usually handles this imbalance better than Logistic Regression without heavy tuning.
    # """)
    
    st.download_button(
        label="📥 Download Report",
        data="Fraud Detection Report\nModel: Random Forest\nStatus: Success",
        file_name="report.txt",
        mime="text/plain"
    )
    
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️")
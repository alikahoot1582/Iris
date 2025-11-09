import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Page configuration
st.set_page_config(
    page_title="Iris Dataset Explorer & Classifier",
    layout="wide"
)

# App title
st.title("🌼 Iris Dataset Explorer & Classifier Project")
st.caption("Created by: Ali Kahoot :) ")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Visualizations", "Model Trainer", "Predict New Sample"])

# Light/Dark mode toggle
theme_choice = st.sidebar.selectbox("Theme", ["Light", "Dark"])

# Set colors dynamically
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .sidebar .sidebar-content {
            background-color: #1E1E1E;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    sns.set_style("darkgrid")
    plt.rcParams.update({
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.facecolor": "#0E1117",
        "figure.facecolor": "#0E1117"
    })
else:
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #FFFFFF;
            color: #000000;
        }
        .sidebar .sidebar-content {
            background-color: #F0F2F6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF"
    })

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    return df

df = load_data()

# ----------------- Dataset Overview -----------------
if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.subheader("Class Distribution")
    st.bar_chart(df["Species"].value_counts())

# ----------------- Visualizations -----------------
elif page == "Visualizations":
    st.header("📈 Visualizations")
    st.subheader("Pairplot")
    fig = sns.pairplot(df, hue="Species")
    st.pyplot(fig)
    st.subheader("Correlation Heatmap")
    fig2, ax = plt.subplots()
    numeric_cols = df.select_dtypes(include="number").columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

# ----------------- Model Trainer -----------------
elif page == "Model Trainer":
    st.header("Train a Random Forest Classifier")
    features = df.columns[1:-1]
    selected_features = st.multiselect("Choose features", features, default=list(features))
    if selected_features:
        X = df[selected_features]
        y = df["Species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        st.warning("Please select at least one feature to train the model.")

# ----------------- Predict New Sample -----------------
elif page == "Predict New Sample":
    st.header("Predict Iris Species from Input Measurements")
    sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
    sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
    petal_length = st.number_input("Petal Length (cm)", value=1.4)
    petal_width = st.number_input("Petal Width (cm)", value=0.2)

    feature_cols = df.columns[1:-1]
    X = df[feature_cols]
    y = df["Species"]
    model = RandomForestClassifier()
    model.fit(X, y)

    if st.button("Predict Species"):
        new_sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_cols)
        prediction = model.predict(new_sample)
        st.success(f"The predicted species is: {prediction[0]}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from inference import predict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"


primary = pd.read_csv(DATA_PATH/"primary_cohort_clean.csv")
study = pd.read_csv(DATA_PATH/"study_cohort_clean.csv")
validation = pd.read_csv(DATA_PATH/"validation_cohort_clean.csv")
df = pd.concat([primary, study, validation], ignore_index=True)

# graph1

fig, ax = plt.subplots()
outcome_pct = (
    df['hospital_outcome_1alive_0dead']
    .value_counts(normalize=True) * 100
)

outcome_pct.plot(kind='bar', ax=ax)
ax.set_xlabel("Hospital outcome")
ax.set_ylabel("Pecentage of patients (%)")
ax.set_title("Survival vs. Mortality Distribution (percentage)")

ax.set_xticklabels(["Dead", "Alive"], rotation=0)

# graph2




app_mode = st.sidebar.selectbox('Select Page', ['Home','Prediction'])

if app_mode == 'Home':
    st.title("Sepsis Survival Analysis")
    st.markdown('### Dataset :')
    st.write(df.head())
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of patients", len(df))
        col2.metric("Average age", round(df["age_years"].mean(), 1))
        col3.metric(
            "Survival rate",
            f"{df['hospital_outcome_1alive_0dead'].mean() * 100:.1f}%"
        )
    
    st.markdown('### Comparison of survivors vs. non-survivors')
    st.pyplot(fig) # graph1
    
    st.markdown('### Anomaly detection :')
    st.metric("Extreme values detected (process using winsorization)", 134) # anomalies


if app_mode == 'Prediction':
    st.title("Sepsis Survival Prediction")
    age = st.slider("Age", min_value=0, max_value=100, value=50)
    sex = st.selectbox("Gender", ["Male", "Female"])
    episode = st.number_input("Episode number", min_value=1, max_value=5)
    input_data = {
        "age_years": age,
        "sex_0male_1female": 1 if sex == "Female" else 0,
        "episode_number": episode
    }
    if st.button("Predict"):
        result = predict(input_data)
        st.write("Prediction:", result["prediction"])
        st.write("Survival probability:", result["probability"])

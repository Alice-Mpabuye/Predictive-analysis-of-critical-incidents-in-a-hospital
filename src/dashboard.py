import streamlit as st
import pandas as pd
from inference import predict, IQRWinsorizer

import __main__
__main__.IQRWinsorizer = IQRWinsorizer

st.title("Sepsis Survival Prediction")


age = st.slider("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1])  # 0=male, 1=female
episode = st.number_input("Episode number", min_value=1, value=1)


if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age_years": age,
        "sex_0male_1female": sex,
        "episode_number": episode
    }])
    
    result = predict(input_df)
    st.success(f"Predicted outcome: {result[0]}")
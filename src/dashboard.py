import streamlit as st

st.title("Sepsis Survival Prediction")


age = st.slider("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Gender", ["Male", "Female"])
episode = st.number_input("Episode number", min_value=1, max_value=5)

input_data = {
    "age_years": age,
    "sex_0male_1female": 1 if sex == "Female" else 0,
    "episode_number": episode
}



from inference import predict

if st.button("Predict"):
    result = predict(input_data)
    st.write("Prediction:", result["prediction"])
    st.write("Survival probability:", result["probability"])

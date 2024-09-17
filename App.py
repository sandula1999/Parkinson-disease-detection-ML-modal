import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model from the pickle file
with open('Parkinsons_Disease.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the scaler
with open('Parkinsons_Disease_Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Main Streamlit app
def main():
    # Title of the app with custom styling
    st.markdown('<h1 style="color: #008080; font-size: 32px;">Parkinson\'s Disease Detection</h1>', unsafe_allow_html=True)

    # Input form for user input
    st.header('User Input')
    # Allow user to input values for each feature
    mdvp_fo = st.text_input("MDVP:Fo(Hz)", value='197.07600')
    mdvp_fhi = st.text_input("MDVP:Fhi(Hz)", value='206.89600')
    mdvp_flo = st.text_input("MDVP:Flo(Hz)", value='192.05500')
    mdvp_jitter_perc = st.text_input("MDVP:Jitter(%)", value='0.00289')
    mdvp_jitter_abs = st.text_input("MDVP:Jitter(Abs)", value='0.00001')
    mdvp_rap = st.text_input("MDVP:RAP", value='0.00166')
    mdvp_ppq = st.text_input("MDVP:PPQ", value='0.00168')
    jitter_ddp = st.text_input("Jitter:DDP", value='0.00498')
    mdvp_shimmer = st.text_input("MDVP:Shimmer", value='0.01098')
    mdvp_shimmer_db = st.text_input("MDVP:Shimmer(dB)", value='0.097')
    shimmer_apq3 = st.text_input("Shimmer:APQ3", value='0.00563')
    shimmer_apq5 = st.text_input("Shimmer:APQ5", value='0.00680')
    mdvp_apq = st.text_input("MDVP:APQ", value='0.00802')
    shimmer_dda = st.text_input("Shimmer:DDA", value='0.01689')
    nhr = st.text_input("NHR", value='0.00339')
    hnr = st.text_input("HNR", value='26.775')
    rpde = st.text_input("RPDE", value='0.422229')
    dfa = st.text_input("DFA", value='0.741367')
    spread1 = st.text_input("spread1", value='-7.3483')
    spread2 = st.text_input("spread2", value='0.177551')
    d2 = st.text_input("D2", value='1.743867')
    ppe = st.text_input("PPE", value='0.085569')

    # Standardize the input data
    input_data = np.array([[float(mdvp_fo), float(mdvp_fhi), float(mdvp_flo), float(mdvp_jitter_perc), 
                            float(mdvp_jitter_abs), float(mdvp_rap), float(mdvp_ppq), float(jitter_ddp), 
                            float(mdvp_shimmer), float(mdvp_shimmer_db), float(shimmer_apq3), float(shimmer_apq5), 
                            float(mdvp_apq), float(shimmer_dda), float(nhr), float(hnr), float(rpde), float(dfa), 
                            float(spread1), float(spread2), float(d2), float(ppe)]])
    std_data = scaler.transform(input_data)

    # Create a button to make predictions
    if st.button("Predict"):
        # Make prediction
        prediction = loaded_model.predict(std_data)
        # Display prediction result with custom styling
        if prediction == 0:
            st.markdown('<h2 style="color: #008000;">The person does not have Parkinson\'s Disease.</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="color: #FF0000;">The person has Parkinson\'s Disease.</h2>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()

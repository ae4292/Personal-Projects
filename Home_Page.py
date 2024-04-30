import streamlit as st

st.set_page_config(
    page_title="Crop Prediction Dashboard"
)
st.markdown("<h1 style='text-align: center; color: black;'>Crop Yield and Land Prediction Project</h1>", unsafe_allow_html=True)
st.markdown('>This interactive dashboard contains three separate models, which can all be found in the sidebar to the left. \
            The first contains a Yield Prediction Model for a given US state, yaer, and other features. The second model focuses \
            on predicting whether a given geographical area in the US might be viable for growing crops. Our final \
            model focuses on predicting what the most suitable crop is for a given geographical area.')
st.markdown(" >If you have any questions or feedback, feel free to reach out to: adrianenders4292@berkeley.edu")

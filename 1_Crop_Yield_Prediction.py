import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import statsmodels.api as sm
import streamlit as st
from joblib import load


model_lc = load('models/model_lc.joblib')
model_ltc = load('models/model_ltc.joblib')
model_rfc = load('models/model_rfc.joblib')
model_rftc = load('models/model_rftc.joblib')
model_xgc = load('models/model_xgc.joblib')
model_xgtc = load('models/model_xgtc.joblib')


value_dict = {'const':[1], 'Year':[0], 'Acres_Harvested':[0], 'Price_in_CWT':[0], 'HUMIDITY_%':[0],
       'PH_soil_type':[0], 'Population':[0], 'Commodity_BARLEY':[0], 'Commodity_CORN':[0],
       'Commodity_LENTILS':[0], 'Commodity_OATS':[0], 'Commodity_PEANUTS':[0],
       'Commodity_RICE':[0], 'Commodity_SUNFLOWER':[0], 'Commodity_WHEAT':[0],
       'State_ALABAMA':[0], 'State_ALASKA':[0], 'State_ARIZONA':[0], 'State_ARKANSAS':[0],
       'State_CALIFORNIA':[0], 'State_COLORADO':[0], 'State_DELAWARE':[0], 'State_FLORIDA':[0],
       'State_GEORGIA':[0], 'State_IDAHO':[0], 'State_ILLINOIS':[0], 'State_INDIANA':[0],
       'State_IOWA':[0], 'State_KANSAS':[0], 'State_KENTUCKY':[0], 'State_LOUISIANA':[0],
       'State_MAINE':[0], 'State_MARYLAND':[0], 'State_MICHIGAN':[0], 'State_MINNESOTA':[0],
       'State_MISSISSIPPI':[0], 'State_MISSOURI':[0], 'State_MONTANA':[0],
       'State_NEBRASKA':[0], 'State_NEVADA':[0], 'State_NEW JERSEY':[0],
       'State_NEW MEXICO':[0], 'State_NEW YORK':[0], 'State_NORTH CAROLINA':[0],
       'State_NORTH DAKOTA':[0], 'State_OHIO':[0], 'State_OKLAHOMA':[0], 'State_OREGON':[0],
       'State_PENNSYLVANIA':[0], 'State_SOUTH CAROLINA':[0], 'State_SOUTH DAKOTA':[0],
       'State_TENNESSEE':[0], 'State_TEXAS':[0], 'State_UTAH':[0], 'State_VIRGINIA':[0],
       'State_WASHINGTON':[0], 'State_WEST VIRGINIA':[0], 'State_WISCONSIN':[0], 
       'State_WYOMING':[0]
}
x_to_pred = pd.DataFrame.from_dict(value_dict)

st.set_page_config(page_title="Crop Prediction Dashboard")

st.markdown("<h1 style='text-align: center; color: black;'>Crop Yield Prediction</h1>", unsafe_allow_html=True)

st.divider()
st.markdown('>Please provide values for your selected State and Year you are trying to predict your Crop Yield for.')
state = st.selectbox("State", ['','ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS',
       'CALIFORNIA', 'COLORADO', 'DELAWARE', 'FLORIDA',
       'GEORGIA', 'IDAHO', 'ILLINOIS', 'INDIANA',
       'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA',
       'MAINE', 'MARYLAND', 'MICHIGAN', 'MINNESOTA',
       'MISSISSIPPI', 'MISSOURI', 'MONTANA',
       'NEBRASKA', 'NEVADA', 'NEW JERSEY',
       'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA',
       'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON',
       'PENNSYLVANIA', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
       'TENNESSEE', 'TEXAS', 'UTAH', 'VIRGINIA',
       'WASHINGTON', 'WEST VIRGINIA', 'WISCONSIN', 
       'WYOMING'])
commodity = st.selectbox('Crop', ['','BARLEY', 'CORN', 'LENTILS', 'OATS', 'PEANUTS',
       'RICE', 'SUNFLOWER', 'WHEAT'])

year = st.slider('Year', min_value=2010, max_value=2030, step=1)
pH = st.slider('Soil pH Level', min_value=0.0, max_value=14.0,step=0.25)
hum = st.slider("Humidity",min_value=0.0, max_value=100.0, step=0.5)
ah = st.number_input('Acres Harvested ', min_value=0)
price = st.number_input('Price in CWT (Hundredweight)', min_value=0.0)
pop = st.number_input("State Population",min_value=0)
tweets = st.selectbox('Incorporate Tweet Data', ['Yes', 'No'])
if tweets == 'Yes':
    prop_WE = st.slider("Percent of Climate Related Tweets about Weather Extremes",min_value=0.0, max_value=100.0, step=0.1)
x_to_pred.loc[0,'Year'] = year
x_to_pred.loc[0,'Acres_Harvested'] = ah
x_to_pred.loc[0,'Price_in_CWT'] = price
x_to_pred.loc[0,'HUMIDITY_%'] = hum
x_to_pred.loc[0,'PH_soil_type'] = pH
x_to_pred.loc[0,'Population'] = pop

if (commodity != '') & (state != ''):

    x_to_pred.loc[0,f'Commodity_{commodity}'] = 1
    x_to_pred.loc[0,f'State_{state}'] = 1

    lasso = model_lc.predict(x_to_pred)[0]
    rf = model_rfc.predict(x_to_pred)[0]
    xg = model_xgc.predict(x_to_pred)[0]
    if tweets == 'Yes':
            x_to_pred['prop_topic_WE']= prop_WE
            lasso = model_ltc.predict(x_to_pred)[0]
            rf = model_rftc.predict(x_to_pred)[0]
            xg = model_xgtc.predict(x_to_pred)[0]
    st.divider()
    st.markdown(f'### Predicted Crop Yield in {state[0] + state[1:].lower()} in {year}')
    st.markdown(f'#### Lasso Linear Regression: {np.round(lasso,2)} pounds per acre')
    st.markdown(f'#### Random Forest Regression: {np.round(rf,2)} pounds per acre')
    st.markdown(f'#### XGBoost Regression: {np.round(xg,2)} pounds per acre')

    st.divider()
    st.markdown("### Comparison of Regression Models")
    st.image('model_comp.png')
    st.markdown('>While XGBoost was our most accurate model, we want to provide different results from varying models to give a possible range of yields')
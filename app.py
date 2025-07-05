# Import libraries
import streamlit as st 
import joblib
import numpy as np
import pandas as pd     

# Load encoder
encoder = joblib.load('encoder.pkl')

# Load scaler
scaler = joblib.load('scaler.pkl')

# Load model
lr_model = joblib.load('lr_model.pkl')

# Title
st.title('Supply Chain Emissions Predictions')

# Important columns
columns = ['Substance', 'Unit', 'Supply Chain Emission Factors without Margins',
       'Margins of Supply Chain Emission Factors',
       'Supply Chain Emission Factors with Margins',
       'DQ ReliabilityScore of Factors without Margins',
       'DQ TemporalCorrelation of Factors without Margins',
       'DQ GeographicalCorrelation of Factors without Margins',
       'DQ TechnologicalCorrelation of Factors without Margins',
       'DQ DataCollection of Factors without Margins', 'Source']

# Markdown
st.markdown("""
This app predicts **Supply Chain Emission Factors with Margins** based on DQ metrics and other parameters.
""")

# Input form
with st.form('prediction_form'):
    substance = st.selectbox('Substance', ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
    unit = st.selectbox('Unit', ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
    margin = st.number_input('Margins of Supply Chain Emission Factors', format='%.3f')
    supply_w_margin = st.number_input('Supply Chain Emission Factors with Margins', format='%.3f')
    dq_reliability = st.slider('DQ Reliability', 0, 5)
    dq_temporal = st.slider('DQ Temporal', 0, 5)
    dq_geographical = st.slider('DQ Geographical', 0, 5)
    dq_technological = st.slider('DQ Technological', 0, 5)
    dq_data_collection = st.slider('DQ Data Collection', 0, 5)
    source = st.selectbox('Source', ['Commodity', 'Industry'])

    # Submit button to perform predictions
    submit = st.form_submit_button('Predict')

# If submit button is clicked
if submit:
    # Data dictionary
    data = {
        'Substance': substance,
        'Unit': unit,
        'Margins of Supply Chain Emission Factors': margin,
        'Supply Chain Emission Factors with Margins': supply_w_margin,
        'DQ ReliabilityScore of Factors without Margins': dq_reliability,
        'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
        'DQ GeographicalCorrelation of Factors without Margins': dq_geographical,
        'DQ TechnologicalCorrelation of Factors without Margins': dq_technological,
        'DQ DataCollection of Factors without Margins': dq_data_collection,
        'Source': source
    }

    # Convert into dataframe
    df = pd.DataFrame([data])

    # Categorical columns
    categorical_columns = ['Substance', 'Unit', 'Source']

    # Encoding
    one_hot_encoded = encoder.transform(df[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded,
                          columns=encoder.get_feature_names_out(categorical_columns))
    df_ohe = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)

    # Scaling
    df_ohe_scaled = scaler.transform(df_ohe)

    # Prediction
    prediction = lr_model.predict(df_ohe_scaled)

    # Output
    st.success(f'Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**')








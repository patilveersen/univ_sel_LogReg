import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
model = joblib.load('admission_model.pkl')
scaler = joblib.load('scaler.pkl')
def predict_admission(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    return 'Yes' if prediction == 1 else 'No'
st.title('Selection to a Univerity Prediction - Siri')
if 'inputs' not in st.session_state:
    st.session_state.inputs = {'gre_score': 0,'toefl_score': 0,'university_rating': 1,'sop': 1,'lor': 1,'cgpa': 0.0,'research': 0}
st.session_state.inputs['gre_score'] = st.number_input('GRE Score', min_value=0, max_value=340, value=st.session_state.inputs['gre_score'])
st.session_state.inputs['toefl_score'] = st.number_input('TOEFL Score', min_value=0, max_value=120, value=st.session_state.inputs['toefl_score'])
st.session_state.inputs['university_rating'] = st.selectbox('University Rating', [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(st.session_state.inputs['university_rating']))
st.session_state.inputs['sop'] = st.slider('Statement of Purpose (SOP)', 1, 5, value=st.session_state.inputs['sop'])
st.session_state.inputs['lor'] = st.slider('Letter of Recommendation (LOR)', 1, 5, value=st.session_state.inputs['lor'])
st.session_state.inputs['cgpa'] = st.number_input('CGPA', min_value=0.0, max_value=10.0, step=0.01, value=st.session_state.inputs['cgpa'])
st.session_state.inputs['research'] = st.selectbox('Research Experience', [0, 1], index=[0, 1].index(st.session_state.inputs['research']))
if st.button('Predict'):
    features = [st.session_state.inputs['gre_score'],st.session_state.inputs['toefl_score'],st.session_state.inputs['university_rating'],st.session_state.inputs['sop'],st.session_state.inputs['lor'],st.session_state.inputs['cgpa'],st.session_state.inputs['research']]
    result = predict_admission(features)
    st.write(f'Chance of Admission: {result}')

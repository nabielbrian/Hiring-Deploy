# === Import libraries ===
import streamlit as st
import home
import eda
import predict as predict

# === Page configuration ===
st.set_page_config(
    page_title="CitoConnect Hiring Decision Support Platform",
    page_icon=":briefcase:"
)

# === Sidebar ===
with st.sidebar:
    st.write("# Page Navigation")

    # Page selection
    page = st.selectbox("Select Page", ("Home", "EDA", 'Predict Hiring and Rating'))

    st.write(f'You are in: {page} page')

    st.write('## About')
    '''
    The **CitoConnect Hiring Decision Support Platform** helps companies make **data-driven hiring decisions**.  
    It predicts a candidate’s hiring outcome, calculates an overall score, and highlights the most important factors.  

    Designed to improve efficiency, reduce bias, and provide fair, explainable results,  
    CitoConnect empowers HR teams to hire **smarter and faster**.
    '''

# === Main content based on selection ===
if page == 'Home':
    home.run()
elif page == 'EDA':
    eda.run()
else:
    predict.run()

import streamlit as st

def run():
    st.markdown("#  CitoConnect Hiring Decision Support Platform")
    
    # Image below the title
    st.image("Training-the-HR-Team-1.20.20.jpg",  use_container_width=True)
    
    st.markdown("""

    This platform is designed for companies to **input candidate data** and receive:
    - **Predicted hiring decision** (Hire / No Hire)
    - **Candidate score** (overall ranking metric)

    Our goal: **Help organizations make rational, data-driven hiring decisions** instead of relying solely on gut feeling.

    ---

    ## Background
    In recruitment, talent acquisition teams must decide which candidates are the best fit.  
    Hiring decisions are often **subjective**, relying on:
    - Interviews
    - Test scores
    - Other qualitative impressions

    However, research shows decisions are stronger when supported by **empirical data**  
    [(*Source: Hiring Decision Support System Research*)](https://arxiv.org/abs/2003.11591)

    At **CitoConnect** — a firm specializing in **data-driven hiring** — we are launching an **improved machine learning platform** to:
    - Classify candidates more effectively
    - Identify those with the best skill sets for specific roles
    - Maintain a competitive edge in an evolving tech landscape

    ---

    ## Problem Statement
    We aim to design:
    - A **predictive ML model**
    - A **candidate scoring formula**

    This system will:
    - Support hiring decisions by identifying candidates most likely to succeed
    - Improve recruitment efficiency
    - Reduce bias
    - Enhance overall talent quality

    ---

    ## Problem Exploration
    1. How many applicants are hired?
    2. What is the average candidate score?
    3. Which characteristics matter most for decision-making?
    4. Does applicant demographics influence hiring?
    5. How do technical scores differ between hired and non-hired candidates?
    6. Which recruitment strategies lead to higher hiring rates?

    ---

    ## Objectives
    - **Data Preparation** → Clean, engineer features, handle class imbalance
    - **Model Development** → Test & compare supervised classification algorithms
    - **Optimization** → Tune hyperparameters, analyze false positives/negatives
    - **Scoring System** → Combine technical + portfolio metrics into one candidate score
    - **Deployment** → Provide explainable, scalable, data-driven hiring recommendations

    ---

    ## Chosen Algorithm: XGBoost
    - High predictive accuracy on structured data
    - Robust to overfitting via advanced regularization
    - Efficient and scalable for large datasets

    ---

    ## Closing Statement
    By combining **ML predictions** with **feature-importance–weighted scoring**,  
    CitoConnect delivers an **explainable ranking system** that:
    - Reflects both technical skills and portfolio strength
    - Minimizes overlooked talent through false positive/negative analysis
    - Can be deployed across multiple companies
    - Includes **bias monitoring** and **periodic recalibration** for fairness

    This platform empowers HR teams to make **faster, fairer, and smarter hiring decisions**.
    """)

if __name__ == "__main__":
    run()








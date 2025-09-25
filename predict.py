import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# === Load Model & Params ===
loaded_model = joblib.load("best_xgb_model.pkl")
with open("best_xgb_params.json", "r") as f:
    best_params = json.load(f)

# === Expected Feature Order from Training ===
EXPECTED_FEATURE_ORDER = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany', 'InterviewScore',
    'SkillScore', 'PersonalityScore', 'RecruitmentStrategy'
]

# === Feature Importance ===
feature_importance = {
    'RecruitmentStrategy': 1.613649,
    'EducationLevel': 0.753971,
    'SkillScore': 0.702591,
    'PersonalityScore': 0.691005,
    'InterviewScore': 0.589251,
    'ExperienceYears': 0.521886,
    'Gender': 0.162188,
    'Age': 0.124089,
    'DistanceFromCompany': 0.111405,
    'PreviousCompanies': 0.102680
}
total_importance = sum(feature_importance.values())
feature_weights = {f: imp / total_importance for f, imp in feature_importance.items()}

# === Min/Max Values ===
feature_minmax = {
    'EducationLevel': (1.0, 4.0),
    'ExperienceYears': (0.0, 40.0),
    'PreviousCompanies': (0.0, 5.0),
    'DistanceFromCompany': (1.0 , 100.0),
    'InterviewScore': (0.0, 100.0),
    'SkillScore': (0.0, 100.0),
    'PersonalityScore': (0.0, 100.0),
    'RecruitmentStrategy': (1.0, 3.0),
    'Gender': (0.0, 1.0),
    'Age': (18.0, 65.0)
}

# === Radar chart config ===
labels = ["PortfolioScore", "TechnicalScore", "RecruitmentStrategy", "ExperienceYears", "EducationLevel"]
max_values = [100, 100, 3, 15, 4]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]


def plot_candidate(stats):
    # Normalize scores for consistent plotting (0–100 scale)
    stats_norm = [s / m * 100 for s, m in zip(stats, max_values)]
    stats_norm += stats_norm[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, stats_norm, color="blue", linewidth=0.5)
    ax.fill(angles, stats_norm, color="skyblue", alpha=0.4)

    # Set axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels,fontsize=7)

    # Standard percentage ticks for numeric axes
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])

    # Custom category labels
    category_labels = {
        "EducationLevel": ["Bachelor T1", "Bachelor T2", "Master's", "PhD"],
        "ExperienceYears": ["5 yrs", "10 yrs", "15 yrs"],
        "RecruitmentStrategy": ["Aggressive", "Moderate", "Conservative"]
    }

    for i, label in enumerate(labels):
        if label in category_labels:
            vals = category_labels[label]
            max_val = max_values[i]
            step = max_val / len(vals)
            angle = angles[i]

            for j, txt in enumerate(vals, start=1):
                r = (j * step) / max_val * 100  # convert to percentage scale
                ax.text(
                    angle, r + 5, txt,  # +5 offset for clarity
                    ha='center', va='center',
                    fontsize=5,
                    rotation=0,  # keep horizontal
                    rotation_mode='anchor'
                )

    return fig

def run():
    st.title("Candidate Score Prediction")

    with st.form("candidate_form"):
        recruitment_strategy = st.selectbox(
            "Recruitment Strategy",
            [1, 2, 3],
            format_func=lambda x: {1: "Aggressive", 2: "Moderate", 3: "Conservative"}[x]
        )
        education_level = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Bachelor’s (Type 1)",
                2: "Bachelor’s (Type 2)",
                3: "Master’s",
                4: "PhD"
            }[x],
            index=1
        )
        skill_score = st.slider("Skill Score", 0, 100, 50)
        personality_score = st.slider("Personality Score", 0, 100, 50)
        interview_score = st.slider("Interview Score", 0, 100, 50)
        experience_years = st.slider("Experience Years", 0, 15, 5)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        age = st.slider("Age", 18, 65, 30)
        distance = st.number_input("Distance from Company (km)", min_value=1.0, max_value=60.0, value=10.0)
        previous_companies = st.slider("Previous Companies", 1, 5, 2)
        submit = st.form_submit_button("Predict Score")

    if submit:
        # Create input dataframe
        inf_data = pd.DataFrame([{
            'RecruitmentStrategy': recruitment_strategy,
            'EducationLevel': education_level,
            'SkillScore': skill_score,
            'PersonalityScore': personality_score,
            'InterviewScore': interview_score,
            'ExperienceYears': experience_years,
            'Gender': gender,
            'Age': age,
            'DistanceFromCompany': distance,
            'PreviousCompanies': previous_companies
        }])

        # Ensure the column order matches training
        inf_data = inf_data[EXPECTED_FEATURE_ORDER]

        # Scale features for heuristic scoring
        inf_scaled = pd.DataFrame()
        for feature in feature_importance.keys():
            min_val, max_val = feature_minmax[feature]
            inf_scaled[feature] = (inf_data[feature] - min_val) / (max_val - min_val)
            inf_scaled[feature] = inf_scaled[feature].clip(0, 1)

        # Calculate scores
        portfolio_score = inf_scaled[['EducationLevel', 'ExperienceYears', 'PreviousCompanies']].mean(axis=1) * 100
        technical_score = inf_scaled[['InterviewScore', 'SkillScore', 'PersonalityScore']].mean(axis=1) * 100
        heuristic_score = sum(inf_scaled[f] * w for f, w in feature_weights.items())
        model_score = loaded_model.predict_proba(inf_data)[:, 1]
        candidate_score = (0.5 * model_score + 0.5 * heuristic_score) * 100

        # Predict
        y_pred_loaded = loaded_model.predict(inf_data)
        prediction_label = "Passed" if y_pred_loaded[0] == 1 else "Not Passed"

        # Background highlight colors
        bg_color = "lightgreen" if prediction_label == "Passed" else "#ff9999"

        # Display results
        st.markdown(
            f"""
            <div style="background-color:{bg_color}; padding:10px; border-radius:10px; text-align:center;">
                <h2 style="color:black;">{prediction_label}</h2>
                <p style="font-size:45px; font-weight:bold;">Candidate Score: {candidate_score[0]:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Radar chart 
        stats = [
            portfolio_score[0],
            technical_score[0],
            recruitment_strategy,
            experience_years,
            education_level
        ]
        fig = plot_candidate(stats)
        st.pyplot(fig)

if __name__ == "__main__":
    run()

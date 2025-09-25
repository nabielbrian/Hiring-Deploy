import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import ttest_ind, chi2_contingency
import joblib
import json

# ===== LOAD DATA AND MODEL =====
@st.cache_data
def load_data():
    return pd.read_csv("recruitment_data.csv")  # replace with your actual dataset file

@st.cache_resource
def load_model():
    model = joblib.load("best_xgb_model.pkl")
    return model

df_raw = load_data()
df_copy = df_raw.copy()
best_xgb = load_model()

# If Gender is numeric, map it to Male/Female
if df_copy['Gender'].dtype in [np.int64, np.int32, np.float64]:
    df_copy['Gender'] = df_copy['Gender'].map({1: "Male", 0: "Female"})

# Features for SHAP
X = df_copy[['Age', 'Gender', 'EducationLevel', 'ExperienceYears',
             'PreviousCompanies', 'DistanceFromCompany', 'InterviewScore',
             'SkillScore', 'PersonalityScore', 'RecruitmentStrategy']]

# ===== QUESTION FUNCTIONS =====
def q1_hired_vs_not_hired():
    counts = df_copy['HiringDecision'].value_counts().sort_index()
    labels = ['Not Hired', 'Hired']
    colors = ['#F44336', '#4CAF50']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(width=0.4), textprops={'fontsize': 12})
    ax.set_title('Applicants Hired vs Not Hired', fontsize=14)
    st.pyplot(fig)
    st.markdown("""
### Applicants Hired vs Not Hired
---
**Visual Assessment**

- Out of all applicants, **31% were hired** while **69% were not hired**.  
- This shows a **more selective hiring process**, where roughly **2 out of every 3 applicants** are rejected.  
- While not surprising in competitive recruitment, the imbalance highlights that most applicants do not pass the final selection stage.
---
**Interpretation:**  
This is a typical funnel pattern in hiring, where a larger applicant pool gets narrowed down to a smaller group of selected candidates. Further analysis is needed to understand **why** certain candidates are hired and others are not — e.g., based on skills, experience, or assessment scores.
""")


def q2_avg_candidate_value():
    labels = ["Interview Score", "Skill Score", "Personality Score", "Experience Years", "EducationLevel"]
    stats = [51, 51, 49, 8, 2]
    max_values = [100, 100, 100, 15, 4]
    stats = [s / m * 100 for s, m in zip(stats, max_values)]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats += stats[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, color="blue", linewidth=2)
    ax.fill(angles, stats, color="skyblue", alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    st.pyplot(fig)
    st.markdown("""
### Average Candidate Profile 

| Field                         | Value (Average Candidate)                             |
| ----------------------------- | ----------------------------------------------------- |
| **Name**                      | Jane/John Doe                                         |
| **Gender**                    | Female (≈49%) / Male (≈51%) → example shown: Male     |
| **Age**                       | 35 years                                              |
| **Education Level**           | Bachelor’s (Type 2)                                   |
| **Experience Years**          | 8 years                                               |
| **Previous Companies Worked** | 3                                                     |
| **Distance From Company**     | 25 km                                                 |
| **Interview Score**           | 51                                                    |
| **Skill Score**               | 51                                                    |
| **Personality Score**         | 49                                                    |
| **Recruitment Strategy**      | Moderate                                              |
| **Hiring Decision**           | Not hired (69% of cases)                              |
---
**Interpretation:**  
The "average" applicant is a mid-career professional with solid educational credentials, a balanced skill and personality profile, and moderate interview performance.  
Despite meeting reasonable standards, the majority of such candidates **do not get hired**, suggesting that hiring decisions may rely on factors beyond these core metrics — possibly **specific skill match, cultural fit, or competitive benchmarks**.
""")


def q3_feature_importance():
    # Copy to avoid modifying the original
    X_copy = X.copy()

    # Encode categorical columns
    for col in X_copy.select_dtypes(include=['object']).columns:
        X_copy[col] = pd.factorize(X_copy[col])[0]  # simple label encoding

    # Ensure numeric dtype
    X_copy = X_copy.apply(pd.to_numeric, errors='coerce').fillna(0)

    explainer = shap.Explainer(best_xgb, X_copy)
    shap_values = explainer(X_copy)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_copy, show=False)
    st.pyplot(fig)
    st.markdown("""
###  SHAP Feature Impact 
---

**Visual Assessment**

**1. Key Drivers of Hiring Decision**  
- **Recruitment Strategy** is the dominant driver (**mean SHAP: 1.61**), far exceeding other features.  
  - Higher values (red points) strongly push predictions towards hiring.  
  - Lower values (blue points) tend to decrease hiring likelihood.  
- **Education Level** (0.75), **Skill Score** (0.70), and **Personality Score** (0.69) also have a noticeable, positive influence.  
  - Red (high feature value) generally corresponds to higher SHAP values → more hiring likelihood.  

**2. Moderate Influencers**  
- **Interview Score** (0.59) and **Experience Years** (0.52) play secondary but still relevant roles.  
  - High scores in either tend to slightly increase hiring probability.  

**3. Minimal Impact Features**  
- **Gender** (0.16), **Age** (0.12), **Distance from Company** (0.11), and **Previous Companies** (0.10) barely influence the model’s decision.  
  - Their SHAP points cluster around zero, indicating neutrality.  

**4. Bias Perspective**  
- Gender and age have **low mean SHAP values**, suggesting the model’s decision-making is **largely independent** of these demographic variables.  
- This aligns with earlier statistical findings showing no significant correlation between demographics and hiring outcomes.

---

**Interpretation:**  
The hiring model prioritizes **strategic recruitment channel**, **education**, and **skills/personality fit** over demographics or logistical details.  
For optimization, focus on:
- Enhancing candidate skill scores  
- Improving recruitment targeting  
- Elevating education-related credentials  
""")


def q4_demographics():
    df_copy['Age'] = pd.to_numeric(df_copy['Age'], errors='coerce')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="HiringDecision", y="Age", data=df_copy, ax=axes[0])
    gender_prop = pd.crosstab(df_copy['HiringDecision'], df_copy['Gender'], normalize='index')
    gender_prop.plot(kind='bar', stacked=True, color=["#66c2a5", "#fc8d62"], ax=axes[1])
    axes[1].set_ylabel("Proportion")
    st.pyplot(fig)
    # Add explanatory text below chart
    st.markdown("""
### Demographic Distribution By Hiring Decision
---
**Visual Assessment**

- **Age Distribution by Hiring Decision** → The median, interquartile range, and spread look very similar for both "Hired" and "Not Hired" groups. No noticeable skew toward younger or older candidates in either outcome.

- **Gender Distribution by Hiring Decision** → The proportions of male (0) and female (1) applicants look nearly identical across both hiring outcomes.

---

**Interpretation:**

The hiring decision appears independent of both age and gender — at least in this dataset. There’s no visible bias toward older/younger candidates or toward one gender over the other.

---

**Age (continuous)**

- **p-value = 0.9429** → Much greater than 0.05  
- This means there is **no statistically significant difference** in the average age of hired vs. not hired candidates.  
- In plain terms: Age does **not** appear to influence hiring decisions in this dataset.

**Gender (categorical)**

- **p-value = 0.9751** from Chi-square test → Also much greater than 0.05  
- This means there is **no statistically significant relationship** between gender and hiring decision.  
- Hiring outcomes are distributed almost equally across genders.

---

**Conclusion:**  
Neither **age** nor **gender** shows a significant statistical relationship with hiring decisions. Based on this dataset, the recruitment process appears **demographically neutral** for these two variables.
""")


def q5_technical_scores():
    tech_cols = ['SkillScore', 'PersonalityScore', 'InterviewScore']
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, col in enumerate(tech_cols):
        if col in df_copy.columns:
            sns.boxplot(data=df_copy, x='HiringDecision', y=col, ax=axs[i])
            axs[i].set_title(f'{col} by Hiring Status')
    st.pyplot(fig)
    st.markdown("""
### Boxplot Analysis – Candidate Scores vs. Hiring Decision
---
**Visual Assessment** 
**1. Skill Score**  
- Median skill score is **higher** for hired applicants.  
- Non-hired candidates show a **wider spread toward lower scores** with more low-skill outliers.  
- **Implication:** Higher skill scores significantly increase hiring likelihood.  

**2. Personality Score**  
- Hired applicants have a **notably higher median** personality score.  
- The interquartile range is shifted upward for hired candidates, highlighting personality as a **strong differentiator**.  

**3. Interview Score**  
- Median interview scores are **consistently higher** for hired applicants.  
- Clear upward shift in distribution suggests **interview performance strongly influences** hiring outcomes.  

---

**Conclusion:**  
High values in **Skill Score**, **Personality Score**, and **Interview Score** are all associated with a greater likelihood of being hired.  
- Personality and interview scores show **particularly large median differences**, indicating that **soft skills and interview performance** may carry substantial weight in final hiring decisions.  
""")

def q6_recruitment_strategy():
    recruitment_summary = df_copy.groupby('RecruitmentStrategy')['HiringDecision'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=recruitment_summary.index, y=recruitment_summary.values, ax=ax)
    ax.set_title('Hiring Rate by Recruitment Strategy')
    st.pyplot(fig)
    st.markdown("""
### Hiring Rate by Recruitment Strategy
---
**Visual Assessment** 
                
**1. Recruitment Strategy 1**  
- Achieves a hiring rate of **~72%**, far surpassing other strategies.  
- Indicates **strong alignment** between candidate quality and organizational hiring criteria.  

**2. Recruitment Strategies 2 & 3**  
- Both show hiring rates of **~13–14%**, much lower than Strategy 1.  
- May attract **lower-quality applicants** or apply **less effective screening** processes.  

---

**Conclusion:**  
Recruitment Strategy **1** is clearly the most effective, delivering a substantially higher proportion of successful hires.  
Strategies **2** and **3** should be reviewed for potential process improvements or better targeting of high-quality candidates.
""")


# ===== STREAMLIT APP =====
def run():
    st.title("Exploratory Data Analysis")
    question_map = {
        "1. How many applicants are hired?": q1_hired_vs_not_hired,
        "2. What is the Average candidate value?": q2_avg_candidate_value,
        "3. Which characteristics are most relevant for decision-making?": q3_feature_importance,
        "4. Does applicants demographic status play a role?": q4_demographics,
        "5. How do technical assessment scores differ?": q5_technical_scores,
        "6. Are there recruitment strategies with higher hiring rates?": q6_recruitment_strategy
    }
    selected_question = st.selectbox("Select an EDA question", list(question_map.keys()))
    question_map[selected_question]()

if __name__ == "__main__":
    run()

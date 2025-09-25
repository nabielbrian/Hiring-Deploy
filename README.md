---

# 💼 CitoConnect — Hiring Decision Support Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-orange.svg)
![Model](https://img.shields.io/badge/Model-XGBoost-blue.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success.svg)

🔗 [Live Dashboard](https://citoconnect.streamlit.app/)

---

## Overview

Hiring decisions are often made subjectively through interviews, test scores, and personal impressions.
Without data-driven support, companies risk **bias, inefficiency, and missed talent opportunities**.

**CitoConnect** is an **ML-powered hiring decision platform** that:

* Predicts hiring outcomes with an **XGBoost classifier**
* Provides **candidate scoring** that combines hard & soft skills
* Offers **visual insights** into recruitment data
* Ensures **fair and consistent decision-making**

---

## Repository Structure

```
ModelDeploymentGhozy/
│
├── app.py                    # Main Streamlit application 
├── home.py                   # Homepage layout and content
├── eda.py                    # Exploratory Data Analysis module
├── predict.py                # Prediction logic script
├── best_xgb_model.pkl        # Trained XGBoost model
├── best_xgb_params.json      # Optimal model parameters 
├── recruitment_data.csv      # Training dataset
├── requirements.txt          # Python dependencies  
├── Training-the-HR-Team-1.20.20.jpg   # Image for homepage

assets/
│
├── Eda.png        # Screenshot of EDA page
├── Home.png       # Screenshot of Home page
├── Predict.png    # Screenshot of Prediction page

notebooks/
├── P1M2_Ghozy_Reuski.ipynb        # Full project workflow
├── P1M2_Ghozy_Reuski_inf.ipynb    # Inference workflow
├── inference_candidates.csv       # Sample inference dataset  
```

---

## Data

* **Source:** Synthetic dataset inspired by [Hiring Decision Support System research](https://arxiv.org/abs/2003.11591)
* **Size:** \~1,000 rows × 11 features
* **Features:**

  * Age, Gender, EducationLevel, ExperienceYears, PreviousCompanies, DistanceFromCompany
  * InterviewScore, SkillScore, PersonalityScore, RecruitmentStrategy
* **Target:** Hiring Decision (Recommended / Not Recommended)
* **Missing Values:** None

---

## Methodology

1. **Data Cleaning & Preprocessing**
2. **Feature Encoding & Scaling**
3. **Model Training (XGBoost)**
4. **Evaluation & Metrics (F1, ROC-AUC, Classification Report)**
5. **Explainability (SHAP Feature Importance)**
6. **Candidate Score Calculation**
7. **Deployment in Streamlit**

---

## Tech Stack

| **Library**      | **Purpose**                                             |
| ---------------- | ------------------------------------------------------- |
| **pandas**       | Data manipulation & analysis                            |
| **numpy**        | Numerical computations                                  |
| **matplotlib**   | Data visualization (plots, charts)                      |
| **seaborn**      | Statistical data visualization                          |
| **scipy.stats**  | Statistical tests (Chi-Square, Kendall’s Tau, T-Test)   |
| **scikit-learn** | Preprocessing, pipelines, ML models, evaluation metrics |
| **xgboost**      | Gradient boosting model (`XGBClassifier`)               |
| **shap**         | Model interpretability (feature importance)             |
| **joblib**       | Save/load models                                        |
| **streamlit**    | Interactive dashboards                                  |
| **json**         | Store optimal hyperparameters                           |

---

## Screenshots

### Home Page

![CitoConnect Home](assets/Home.png)

### EDA Page

![CitoConnect EDA](assets/Eda.png)

### Prediction Page

![CitoConnect Prediction](assets/Predict.png)

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/yourusername/ModelDeploymentGhozy.git
cd ModelDeploymentGhozy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## Results & Insights

* **XGBoost achieved strong classification performance** with optimized hyperparameters
* **SHAP analysis** revealed that **InterviewScore, SkillScore, and PersonalityScore** were the most influential features in hiring decisions
* **Candidate Scoring System** gives a holistic evaluation, balancing model probability with heuristic weightings
* **Deployment** allows HR teams to interactively test candidates and make more **consistent, data-driven decisions**

---

## Limitations

* Based on synthetic data → may not fully generalize to real-world cases
* Limited to structured attributes (no resumes, interviews, etc.)
* Model designed for **support**, not as a final hiring authority

---

## References

* [Hiring Decision Support System (arXiv)](https://arxiv.org/abs/2003.11591)
* [Bias Monitoring in Recruitment (arXiv)](https://arxiv.org/abs/2309.13933)
* [Multi-Task Weight Optimization (MDPI)](https://www.mdpi.com/2076-3417/15/5/2473)
* [Data-Driven HR (arXiv)](https://arxiv.org/abs/1606.05611)

---

**Author**: Ghozy Reuski

* GitHub: [@GhozyAlfisyahrReuski]([https://github.com/yourusername](https://github.com/GhozyAlfisyahrReuski))
* LinkedIn: [Ghozy Alfisyahr Reuski]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/ghozy-alfisyahr-reuski-1133481ba/))

---

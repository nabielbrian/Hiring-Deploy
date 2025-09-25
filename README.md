# Hiring Decision Support Platform

Platform berbasis **Streamlit** untuk membantu perusahaan mengambil keputusan rekrutmen dengan data.  
Menggabungkan **machine learning (XGBoost)** + **heuristic scoring** → memprediksi hasil seleksi, memberi skor kandidat, dan menyediakan analisis EDA.

---

## 🎯 Objective
- Mengembangkan **ML model** untuk prediksi perekrutan.  
- Memberikan **Candidate Score** (gabungan ML + heuristic).  
- Menyediakan **analisis data interaktif** untuk HR.  
- Membantu proses rekrutmen jadi lebih **efisien & fair**.  

---

## 🏠 Home Page
- Penjelasan latar belakang & tujuan.  
- Problem statement: subjektivitas dalam rekrutmen.  
- Solusi: model XGBoost + scoring system.  
- Closing: fokus pada fairness, explainability, dan deployment.  

---

## 📊 EDA Page
Interaktif, menjawab 6 pertanyaan:
1. Jumlah kandidat yang diterima vs ditolak.  
2. Profil rata-rata kandidat.  
3. Faktor paling relevan (SHAP feature importance).  
4. Apakah demografi (usia, gender) berpengaruh.  
5. Perbedaan skor teknis (skill, interview, personality).  
6. Strategi rekrutmen mana paling efektif.  

---

## 🤖 Prediction Page
- Input kandidat: usia, gender, pendidikan, pengalaman, skor teknis, jarak, strategi rekrutmen, dll.  
- Output:
  - Prediksi **Lulus / Tidak Lulus** seleksi.  
  - **Candidate Score** (gabungan ML + heuristic).  
  - Radar chart visualisasi kekuatan kandidat.  

---

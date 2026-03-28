# 🌙 SleepIQ — Sleep Quality Predictor

## 📌 Project Overview
SleepIQ is a **Machine Learning-based Sleep Quality Predictor** that analyzes lifestyle, behavioral, and physiological factors to predict the quality of a person's sleep. The system takes daily inputs such as screen time, exercise, caffeine intake, and stress levels to classify sleep quality as **Good**, **Average**, or **Poor**, and provides personalized recommendations to improve sleep health.

---

## 🎯 Objectives
- To predict sleep quality based on real-world lifestyle and health data.
- To provide personalized, actionable suggestions for improving sleep habits.
- To build a fully working local web app using Flask and Machine Learning.

---

## ✨ Key Features
- 🔮 **ML Prediction** — Random Forest model classifies sleep as Good / Average / Poor
- 💡 **Smart Suggestions** — Rule-based tips based on screen time, stress, exercise, caffeine
- 📊 **Analytics Page** — Model accuracy, confusion matrix, per-class metrics, feature importance
- 📅 **History Page** — Tracks all past predictions with a score trend chart
- 🌐 **Web Interface** — Clean, responsive dark-themed UI built with Flask + HTML/CSS
- 💾 **Model Persistence** — Trained model saved as `model.pkl` for fast reload

---

## 🏗 System Architecture

```
User Input (Web Form)
        │
        ▼
Flask Backend (app.py)
        │
        ▼
Random Forest Model (model.pkl)
        │
        ▼
Prediction + Suggestion Engine
        │
        ▼
Web UI — Predictor / Analytics / History
```

---

## 🛠 Technology Stack

| Component         | Technology Used         |
|-------------------|------------------------|
| Programming Language | Python               |
| Web Framework     | Flask                  |
| ML Algorithm      | Random Forest (scikit-learn) |
| Data Processing   | pandas, NumPy          |
| Frontend          | HTML, CSS, JavaScript  |
| Model Storage     | pickle (model.pkl)     |
| Dataset           | Sleep Health & Lifestyle Dataset (Kaggle) |

---

## 📂 Project Structure

```
sleep-predictor/
│
├── app.py                  # Flask backend + ML model training & prediction
├── model.pkl               # Saved trained Random Forest model
├── templates/
│     └── index.html        # Frontend UI (Predictor, Analytics, History)
├── static/
│     └── style.css         # Dark theme styling
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙ Installation and Setup

**Step 1: Clone the repository**
```bash
git clone https://github.com/your-username/sleepiq-predictor.git
cd sleepiq-predictor
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: (Optional) Add Kaggle Dataset**
- Download from: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
- Place `Sleep_health_and_lifestyle_dataset.csv` in the project folder
- In `app.py`, change `load_artifact()` to `train_from_csv("Sleep_health_and_lifestyle_dataset.csv")` once, run, then revert

**Step 4: Run the app**
```bash
python app.py
```

**Step 5: Open in browser**
```
http://127.0.0.1:5000
```

---

## 📊 Pages & Features

| Page        | Features |
|-------------|----------|
| 🔮 Predictor | Input form, ML prediction, score ring, personalized tips |
| 📊 Analytics | Accuracy stats, confusion matrix, per-class F1/precision/recall, feature importance |
| 📅 History   | Past predictions table, sleep score trend chart |

---

## 🧠 ML Model Details

- **Algorithm:** Random Forest Classifier (200 trees)
- **Input Features:** Sleep Duration, Stress Level, Exercise, Screen Time, Caffeine, Mood, Interruptions, Bedtime Hour
- **Output Classes:** Good / Average / Poor
- **Trained on:** Sleep Health and Lifestyle Dataset (Kaggle)

---

## 💡 Suggestion Engine

| Condition | Tip |
|-----------|-----|
| Screen time > 30 min | Reduce screen usage before bed |
| Exercise < 30 min | Try at least 30 mins of physical activity |
| Stress > 6/10 | Practice relaxation techniques |
| Caffeine ≥ 2 cups | Avoid caffeine after 2 PM |
| Sleep < 7 hours | Aim for 7–9 hours of sleep |

---

## ⚠ Limitations
- Predictions depend on quality and size of training data
- Rule-based suggestions do not replace medical advice
- Requires local server to run (not deployed online)
- Session history is cleared when the server restarts

---

## 🚀 Future Improvements
- Deploy on cloud (Render / Railway / Heroku)
- Add user login and persistent history with database
- Integrate real Kaggle dataset for better accuracy
- Add sleep diary NLP analysis
- Mobile-responsive PWA version

---

## 📌 Project Impact
This project demonstrates skills in **Machine Learning**, **Flask web development**, **data preprocessing**, and **UI design** — making it suitable as a mini project, resume project, or academic submission in the domains of health analytics and applied ML.

---

## 📎 Dataset Reference
- [Sleep Health and Lifestyle Dataset — Kaggle](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)

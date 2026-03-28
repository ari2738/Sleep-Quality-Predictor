"""
SleepIQ - Sleep Quality Predictor
Flask + Random Forest | Fixed prediction + History + Analytics
Run: python app.py  →  http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import os, pickle, json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)
app.secret_key = "sleepiq_secret_2024"

MODEL_PATH = "model.pkl"

# ── Suggestion engine ──────────────────────────────
def get_suggestions(d):
    tips = []
    if d["screen_time"] > 30:
        tips.append("📵 High screen time — Reduce screen usage to under 30 min before bed.")
    if d["exercise"] < 30:
        tips.append("🏃 Low exercise — Try at least 30 mins of physical activity daily.")
    if d["stress"] > 6:
        tips.append("🧘 High stress — Practice relaxation techniques like deep breathing or meditation.")
    if d["caffeine"] >= 2:
        tips.append("☕ High caffeine — Avoid caffeine after 2 PM, it disrupts deep sleep.")
    if d["sleep_hours"] < 7:
        tips.append("⏰ Low sleep duration — Aim for 7–9 hours for full recovery.")
    if d["interruptions"] == 1:
        tips.append("🌡️ Sleep interruptions — Check noise/temperature or consult a doctor.")
    if d["mood"] <= 2:
        tips.append("😌 Low mood — Try journaling or gratitude practice before bed.")
    if not tips:
        tips.append("⭐ Excellent habits — Keep your consistent sleep schedule going!")
    return tips[:4]

# ── Build / Load model ─────────────────────────────
def build_model():
    np.random.seed(42)
    n = 400  # 400 samples per class = 1200 total, perfectly balanced

    def make_class(label, n):
        if label == "Good":
            sleep  = np.random.uniform(7, 9, n)
            stress = np.random.randint(1, 5, n)
            exer   = np.random.uniform(30, 90, n)
            screen = np.random.uniform(0, 30, n)
            caf    = np.random.choice([0, 1], n, p=[0.6, 0.4])
            mood   = np.random.randint(3, 6, n)
            intr   = np.zeros(n, dtype=int)
            bed    = np.random.uniform(21, 23, n)
        elif label == "Average":
            sleep  = np.random.uniform(6, 8, n)
            stress = np.random.randint(4, 7, n)
            exer   = np.random.uniform(10, 40, n)
            screen = np.random.uniform(30, 90, n)
            caf    = np.random.choice([1, 2], n, p=[0.5, 0.5])
            mood   = np.random.randint(2, 4, n)
            intr   = np.random.choice([0, 1], n, p=[0.5, 0.5])
            bed    = np.random.uniform(22, 25, n)
        else:  # Poor
            sleep  = np.random.uniform(3, 6, n)
            stress = np.random.randint(7, 11, n)
            exer   = np.random.uniform(0, 20, n)
            screen = np.random.uniform(60, 180, n)
            caf    = np.random.choice([2, 3], n, p=[0.4, 0.6])
            mood   = np.random.randint(1, 3, n)
            intr   = np.ones(n, dtype=int)
            bed    = np.random.uniform(1, 4, n)
        return np.column_stack([sleep, stress, exer, screen, caf, mood, intr, bed])

    X_good = make_class("Good", n)
    X_avg  = make_class("Average", n)
    X_poor = make_class("Poor", n)

    X      = np.vstack([X_good, X_avg, X_poor])
    labels = np.array(["Good"]*n + ["Average"]*n + ["Poor"]*n)

    print("Class distribution:", dict(zip(*np.unique(labels, return_counts=True))))

    le     = LabelEncoder()
    y      = le.fit_transform(labels)
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                   min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test),
                                   target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, model.predict(X_test)).tolist()

    print(f"✅ Random Forest trained — Accuracy: {acc:.3f}")
    print(f"   Classes: {le.classes_}")

    artifact = dict(model=model, le=le, scaler=scaler, acc=acc,
                    report=report, cm=cm, classes=list(le.classes_))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    return artifact

def load_artifact():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            a = pickle.load(f)
        print(f"✅ Model loaded — Acc: {a['acc']:.3f}")
        return a
    return build_model()

artifact = load_artifact()

# ── Routes ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           accuracy=round(artifact["acc"]*100, 1),
                           page="predictor")

@app.route("/history")
def history():
    hist = session.get("history", [])
    return render_template("index.html",
                           accuracy=round(artifact["acc"]*100, 1),
                           page="history",
                           history=hist)

@app.route("/analytics")
def analytics():
    report  = artifact.get("report", {})
    cm      = artifact.get("cm", [])
    classes = artifact.get("classes", [])
    acc     = round(artifact["acc"]*100, 1)

    # Per-class metrics for chart
    metrics = []
    for cls in classes:
        if cls in report:
            metrics.append({
                "name": cls,
                "precision": round(report[cls]["precision"]*100, 1),
                "recall":    round(report[cls]["recall"]*100, 1),
                "f1":        round(report[cls]["f1-score"]*100, 1),
            })

    return render_template("index.html",
                           accuracy=acc,
                           page="analytics",
                           metrics=metrics,
                           cm=cm,
                           classes=classes,
                           weighted_f1=round(report.get("weighted avg",{}).get("f1-score",0)*100,1))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400

        bed_hour = 23
        try: bed_hour = int(str(data.get("bedtime","23:00")).split(":")[0])
        except: pass

        inp = {
            "sleep_hours":  float(data.get("sleep_hours") or 7),
            "stress":       int(data.get("stress") or 5),
            "exercise":     float(data.get("exercise") or 30),
            "screen_time":  float(data.get("screen_time") or 60),
            "caffeine":     int(data.get("caffeine") or 1),
            "mood":         int(data.get("mood") or 3),
            "interruptions":int(data.get("interruptions") or 0),
            "bedtime_hour": bed_hour,
        }

        X = np.array([[inp["sleep_hours"], inp["stress"], inp["exercise"],
                       inp["screen_time"], inp["caffeine"], inp["mood"],
                       inp["interruptions"], inp["bedtime_hour"]]])

        model   = artifact["model"]
        le      = artifact["le"]
        pred    = model.predict(X)[0]
        proba   = model.predict_proba(X)[0]
        label   = le.inverse_transform([pred])[0]
        conf    = round(float(proba[pred]) * 100, 1)
        tips    = get_suggestions(inp)

        # Realistic score based on probabilities
        class_list = list(le.classes_)
        score_weights = {"Good": 85, "Average": 55, "Poor": 25}
        score = int(sum(proba[i] * score_weights.get(class_list[i], 50)
                        for i in range(len(class_list))))
        score = max(5, min(99, score))

        # Save to session history
        hist = session.get("history", [])
        hist.insert(0, {
            "date":     datetime.now().strftime("%d %b %Y"),
            "time":     datetime.now().strftime("%I:%M %p"),
            "quality":  label,
            "score":    score,
            "duration": inp["sleep_hours"],
            "stress":   inp["stress"],
            "exercise": inp["exercise"],
            "screen":   inp["screen_time"],
        })
        session["history"] = hist[:20]

        return jsonify({
            "quality":     label,
            "confidence":  conf,
            "score":       score,
            "suggestions": tips,
            "status":      "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session["history"] = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🌙 SleepIQ → http://127.0.0.1:5000")
    app.run(debug=True)

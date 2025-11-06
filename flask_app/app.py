from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import os
from werkzeug.security import generate_password_hash, check_password_hash
from popper import predict_sentiments
app = Flask(__name__)
app.secret_key = "supersecretkey"  # change in production
MODEL_PATH = "sentiment_model_xgb.pkl"
# Load model + label encoder
vectorizer, model, le = joblib.load(MODEL_PATH)

def predict_sentiment(text):
    pips = predict_sentiments(text, vectorizer, model, le)
    top_label = pips[0][0]
    return {"predicted": top_label, "probabilities": pips}

# File paths
USERS_FILE = "users.csv"
FEEDBACK_FILE = "feedback.csv"

# Create files if not exist
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USERS_FILE, index=False)

if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["username", "text", "predicted_label", "user_feedback"]).to_csv(FEEDBACK_FILE, index=False)

# --------------------------
# Utility Functions
# --------------------------
# def predict_sentiment(text):
#     clean = text.lower()
#     vec = vectorizer.transform([clean])
#     probs = model.predict_proba(vec)[0]
#     result = sorted(zip(le.classes_, probs), key=lambda x: x[1], reverse=True)
#     top_label = result[0][0]
#     return {"predicted": top_label, "probabilities": result}

# --------------------------
# Routes
# --------------------------

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = pd.read_csv(USERS_FILE)
        if username in users["username"].values:
            flash("Username already exists!", "error")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        new_user = pd.DataFrame([[username, hashed_pw]], columns=["username", "password"])
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv(USERS_FILE, index=False)
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = pd.read_csv(USERS_FILE)
        user_row = users[users["username"] == username]
        if not user_row.empty and check_password_hash(user_row.iloc[0]["password"], password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# Dashboard
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    probs = None

    if request.method == "POST" and "text_input" in request.form:
        text_input = request.form["text_input"]
        pred = predict_sentiment(text_input)
        prediction = pred["predicted"]
        probs = pred["probabilities"]
        session["last_prediction"] = {
            "text": text_input,
            "predicted": prediction
        }

    if request.method == "POST" and "feedback" in request.form:
        feedback = request.form["feedback"]
        last_pred = session.get("last_prediction")
        if last_pred:
            feedback_entry = pd.DataFrame([{
                "username": session["user"],
                "text": last_pred["text"],
                "predicted_label": last_pred["predicted"],
                "user_feedback": feedback
            }])
            feedback_entry.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
            flash("Feedback recorded successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("dashboard.html", prediction=prediction, probs=probs, username=session["user"])

if __name__ == "__main__":
    app.run(debug=True)

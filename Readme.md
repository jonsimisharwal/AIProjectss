 Project Title:
 User Behavior Analysis & Churn Prediction System using Machine Learning and FastAPI
 Objective:
 To develop a smart API-based system that predicts user churn probability based on real-time
 behavioral metrics like pages visited, time spent, login status, and user activity days. This system
 aids businesses in retaining users by understanding their engagement levels.
 Tools & Technologies Used:- Programming Language: Python- Web Framework: FastAPI- Model Deployment Server: Uvicorn- Libraries: Scikit-learn, NumPy, Pydantic- Model Format: .pkl (Pickle for serialized ML model)- API Testing: Swagger UI (Auto by FastAPI)
 Problem Statement:
 In today's competitive digital landscape, identifying users likely to leave a platform (i.e., churn) is
 crucial. Manual observation is not scalable. An automated model is needed to predict churn based
 on user interaction behavior.
 Goals:- Accept real-time user interaction data via API.- Predict if a user will churn (0 - Not churn, 1 - Likely to churn).
- Return prediction with a confidence score (model probability).- Enable web-based frontend or business applications to use this backend API for real-time
 decision-making.
 Features:- Accepts POST requests with user behavior data.- ML model predicts churn status.- Returns prediction (0 or 1) and confidence (float).- Auto documentation available at http://127.0.0.1:8000/docs
 Input Schema:
 {
 }
  "pages_visited": 12,
  "time_spent": 180.5,
  "days_since_signup": 10,
  "is_logged_in": true
 Output Example:
 {
 }
  "prediction": "1",
  "confidence": 0.82
 Model Training:
 The model was trained using synthetic/collected data including:- pages_visited: Number of pages the user viewed.
- time_spent: Total time (in minutes) on the site.- days_since_signup: Account age.- is_logged_in: Users login status (True/False).
 Used algorithms like:- Logistic Regression (initially)- Can be improved using Random Forest or Gradient Boost
 FastAPI Endpoints:
 | Endpoint   | Method | Description                        |
 |------------|--------|------------------------------------|
 | /predict   | POST   | Takes user data and returns prediction |
 | /docs      | GET    | Swagger UI for API interaction     |
 Practical Usage:- SaaS businesses for user retention strategies.- E-commerce for customer behavior analysis.- Digital platforms to track engagement and forecast drop-off.
 Future Scope:- Add more features like location, referral source, device type.- Use time-series data and LSTM for deeper prediction.- Integrate with frontend dashboards (React/Vue).- Alert-based system for churn-risk users

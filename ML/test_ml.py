import joblib

model = joblib.load("risk_model.pkl")

# Example user input
input_data = [[3, 4, 8, 6, 7, 1, 2]]  
# mood, sleep, stress, study, screen, appetite, social

prediction = model.predict(input_data)[0]

risk_map = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

print("Predicted Risk:", risk_map[prediction])
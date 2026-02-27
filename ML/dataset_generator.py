import pandas as pd
import random

data = []

for _ in range(1000):
    mood = random.randint(1, 10)
    sleep = random.randint(3, 9)
    stress = random.randint(1, 10)
    study = random.randint(0, 10)
    screen = random.randint(1, 12)
    appetite = random.randint(0, 1)
    social = random.randint(0, 10)

    # Base risk logic
    if mood <= 3 and stress >= 8:
        risk = 2
    elif mood <= 5 and stress >= 6:
        risk = 1
    else:
        risk = 0

    # Add 5% noise
    if random.random() < 0.05:
        risk = random.randint(0, 2)

    data.append([mood, sleep, stress, study, screen, appetite, social, risk])

df = pd.DataFrame(data, columns=[
    "mood", "sleep", "stress", "study",
    "screen", "appetite", "social", "risk"
])

print("Generated rows:", len(df))

df.to_csv("mindguard_dataset.csv", index=False)

print("Dataset generated successfully!")
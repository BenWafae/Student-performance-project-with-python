import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Charger dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\student_perf\student-mat.csv', sep=";")

# Encodage
le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col])

# Features / Target
X = df.drop("G3", axis=1)
y = df["G3"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde CORRECTE
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle entraîné et sauvegardé avec succès")

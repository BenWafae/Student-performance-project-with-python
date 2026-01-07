import pandas as pd

#df = pd.read_csv(r'C:\Users\hp\Desktop\student_perf\student-mat.csv', sep=";")

#print(df.shape)
#print(df.head())
#print(df.info())
#df.isnull().sum()
#df = df.drop_duplicates()
#from sklearn.preprocessing import LabelEncoder

#le = LabelEncoder()

#for col in df.select_dtypes(include="object").columns:
  #  df[col] = le.fit_transform(df[col])
#import matplotlib.pyplot as plt
#import seaborn as sns

#sns.histplot(df["G3"], bins=20)
#plt.title("Distribution de la note finale G3")
#plt.show()
#corr = df.corr()

#plt.figure(figsize=(12,8))
#sns.heatmap(corr, cmap="coolwarm")
#plt.title("Matrice de corrélation")
#plt.show()
#y = df["G3"]
#X = df.drop("G3", axis=1)
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(
  #  X, y, test_size=0.2, random_state=42
#)
#from sklearn.ensemble import RandomForestRegressor

#model = RandomForestRegressor(
 #   n_estimators=200,
 #   random_state=42
#)

#model.fit(X_train, y_train)
#from sklearn.metrics import mean_absolute_error, r2_score

#y_pred = model.predict(X_test)

#print("MAE :", mean_absolute_error(y_test, y_pred))
#print("R²  :", r2_score(y_test, y_pred))
df = pd.read_csv(r'C:\Users\hp\Desktop\student_perf\student-mat.csv', sep=";")
df.head()
df.info()
df.describe()
df['Success'] = (df['G3'] >= 10).astype(int)
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})
df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})
df['Average'] = (df['G1'] + df['G2']) / 2
df['RiskLevel'] = df['absences'] + df['failures']
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Success', data=df)
plt.show()
sns.boxplot(x='Success', y='absences', data=df)
plt.show()
from sklearn.model_selection import train_test_split

X = df[['age', 'studytime', 'absences', 'failures', 'Average']]
y = df['Success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

accuracy
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.show()
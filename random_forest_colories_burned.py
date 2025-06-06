import pandas as pd

df = pd.read_csv(r"calories.csv")
df.head(10)

df.drop(['User_ID'], axis=1, inplace=True)
df.head(10)
df.dropna()
df.Gender[df.Gender == 'male'] = 0
df.Gender[df.Gender == 'female'] = 1
df['Gender']= df['Gender'].values.astype('int')
df.head(10)


Y = df['Calories'].values

X = df.drop(['Calories'], axis = 1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2, random_state=42,shuffle=True)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)
from sklearn import metrics

print("r2_train = ", metrics.r2_score(y_train,y_train_pred))

print("r2_test = ", metrics.r2_score(y_test,y_pred))
print("mse = ", metrics.mean_squared_error(y_test, y_pred))

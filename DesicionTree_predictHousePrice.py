import pandas as pd

data = pd.read_csv(r"C:\Users\5030993\Downloads\melb-data\melb_data.csv")
#data.describe()
data.columns
data= data.dropna(axis=0)
#prediction target
y = data.Price
#features
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[features]
x.describe()

from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)
# Fit model
model.fit(x, y)
print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(model.predict(x.head()))

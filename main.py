from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import quandl
from yaml_reader import read_config

config = read_config('config/config.yaml')
api_key = config['api_key']
quandl.ApiConfig.api_key = api_key

data = quandl.get("FRED/DCOILBRENTEU", start_date="2000-01-01", end_date="2020-12-31")

data.head()

plt.ylabel('Crude Oil Prices: Brent - Europe')
data.Value.plot(figsize=(10, 5))

data['moving_ave_3'] = np.round(data['Value'].shift(1).rolling(window=3).mean(), 2)
data['moving_ave_9'] = np.round(data['Value'].shift(1).rolling(window=9).mean(), 2)

data = data.dropna()


X = data[['moving_ave_3', 'moving_ave_9']]
X.head()

Y = data['Value']
Y.head()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_pred = lr.predict(x_test)
lr_score = np.round((lr.score(x_test, y_test)*100), 2)
print("The model has", lr_score, "% accuracy.")


predicted_price = lr.predict(x_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['Predicted Price', 'Actual Price'])
plt.ylabel("Crude Oil Prices: Brent - Europe")
plt.show()


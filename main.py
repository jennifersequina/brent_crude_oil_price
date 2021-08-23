from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import seaborn as sns

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
lr_score = lr.score(x_test, y_test)
print(lr_score)

lr_score_train = lr.score(x_train, y_train)
print(lr_score_train)

predicted_price = lr.predict(x_test)
predicted_price = pd.DataFrame(predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y_test.plot()
plt.legend(['Predicted Price', 'Actual Price'])
plt.ylabel("Crude Oil Prices: Brent - Europe")
plt.show()

# X_addC = sm.add_constant(X)
# result = sm.OLS(Y, X_addC).fit()

Rcross = cross_val_score(lr, X, Y, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is", Rcross.std())

lr_intercept = lr.intercept_
print(lr_intercept)
# 0.11675701049554732

lr_coef = lr.coef_
print(lr_coef)
# [ 1.23507039 -0.23652455]

ax1 = sns.distplot(X, hist=False, color="r", label="Actual Price")
sns.distplot(predicted_price, hist=False, color="b", label="Fitted Values", ax=ax1)
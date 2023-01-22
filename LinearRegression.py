mport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
Happiness_Forecast = pd.read_csv("C:\\Information_Science\\My_projects\\new_world_happiness_report_2021.csv")
print(Happiness_Forecast)
X = Happiness_Forecast.iloc[:, :-1].values

y = Happiness_Forecast.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)
import statsmodels.api as sm
import statsmodels.tools.tools as tl
X = tl.add_constant(X)

SL = 0.05
X_opt = X[:, [0,1,2,3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary())

numVars = len(X_opt[0])
for i in range(0,round(numVars)):
    regressor_OLS = sm.OLS(y,X_opt).fit()
    max_var = max(regressor_OLS.pvalues).astype(float)
    if max_var > SL:
        new_Num_Vars = (X_opt[0])
        for j in range(0,round(new_Num_Vars)):
            if (regressor_OLS.pvalues[j].astype(float)==max_var):
                X_opt = np.delete(X_opt,j,1)
print(regressor_OLS.summary())

print(y_pred)
print(regressor_OLS.pvalues)
print(regressor_OLS.params)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
plt.figure(figsize=(10,8))
plt.scatter(y_test,y_pred)
#X value represents the following variables:
# GDP per capita, healthy life expectance and individual freedom
plt.xlabel('Variables')
plt.ylabel('Happiness Rate')
plt.title('Happiness Rate Prediction')

print(plt.show())
predicted_y = regressor.predict
print(predicted_y)
y_pred = regressor.predict(X_test)

#I found out there is a medium correlation between undependent variables of GDP per capita, Healthy life index and individual freedom , and
#dependent variable of Happiness rank.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

data = pd.read_excel("data.xlsx")

features = ['Consum[MW]', 'Medie Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
            'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']

data['Data'] = pd.to_datetime(data['Data'], errors='coerce')

data['Sold[MW]'] = pd.to_numeric(data['Sold[MW]'], errors='coerce')  # Transformă în numeric, NaN pentru valori invalide

data[features] = data[features].apply(pd.to_numeric, errors='coerce')

data = data.dropna(subset=['Sold[MW]'] + features)

# excludem luna decembrie 2024 pentru antrenament
data_train = data[~((data['Data'].dt.month == 12) & (data['Data'].dt.year == 2024))]
data_test = data[(data['Data'].dt.month == 12) & (data['Data'].dt.year == 2024)]

X_train = data_train[features]
y_train = data_train['Sold[MW]']
X_test = data_test[features]
y_test = data_test['Sold[MW]']

#standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#am ales modelul Random Forest pentru regresie, un inlocuitor ID3
print("\n--- Modelul Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
print(f"Random Forest - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")

#Naiv Bayes 
print("\n--- Modelul Naiv Bayes ---")
bayes_model = GaussianNB()
bayes_model.fit(X_train_scaled, y_train)

# predictii pentru Naiv Bayes
bayes_pred = bayes_model.predict(X_test_scaled)

bayes_rmse = np.sqrt(mean_squared_error(y_test, bayes_pred))
bayes_mae = mean_absolute_error(y_test, bayes_pred)
print(f"Naiv Bayes - RMSE: {bayes_rmse:.2f}, MAE: {bayes_mae:.2f}")


#print("\nRezultate finale:")
#print(f"Random Forest - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")
#print(f"Naiv Bayes - RMSE: {bayes_rmse:.2f}, MAE: {bayes_mae:.2f}")

#soldul pentru decembrie 2024
sold_total_rf = rf_pred.sum()  # Predicție totală pentru decembrie pe baza Random Forest
sold_total_bayes = bayes_pred.sum()  # Predicție totală pentru decembrie pe baza Bayesian

print(f"Sold total estimat Random Forest pentru decembrie 2024: {sold_total_rf:.2f} MW")
print(f"Sold total estimat Naiv Bayes pentru decembrie 2024: {sold_total_bayes:.2f} MW")

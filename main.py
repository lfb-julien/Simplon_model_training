import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import pickle
import os

# Lecture du fichier CSV
def read_csv(file_path):
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

# Calcul de la colonne 'prix_m2'
def calculate_prix_m2(df):
    df['prix_m2'] = df['prix'] / df['surface_habitable']

# Entraînement des modèles
def train_models(X_train, y_train):
    models = {
        'Decision Tree': DecisionTreeRegressor(max_depth=4),
        'K Neighbors': KNeighborsRegressor(n_neighbors=50),
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(max_depth=100, min_samples_leaf=10, n_estimators=1000)
    }

    for model_name, model in tqdm(models.items(), desc="Fitting models"):
        model.fit(X_train, y_train)
        save_model(model, model_name)

# Sauvegarde des modèles
def save_model(model, model_name):
    folder_path = 'models'
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{model_name}.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

# Graphiques avec contours
def plot_with_contour(model, X, y, title):
    fig = plt.figure()
    # ... (code pour les graphiques avec contours)

# Grid Search pour optimiser les hyperparamètres
def grid_search_models(params_grid, X_train, y_train, X_test, y_test):
    results = []

    for model_name, model_config in tqdm(params_grid.items(), desc="Grid Search Progress"):
        gs = GridSearchCV(estimator=model_config['model'], param_grid=model_config['params'], n_jobs=-1)
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        best_params = gs.best_params_
        train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
        score = best_model.score(X_test, y_test)

        results.append({
            "Model": model_name,
            "Optimal Params": str(best_params),
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Model Score": score
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

# Exécution du code
if __name__ == "__main__":
    # Lecture du fichier CSV
    df = read_csv('transactions.csv')

    # Calcul de la colonne 'prix_m2'
    calculate_prix_m2(df)

    # Filtrage pour les transactions à Paris en 2022
    paris_df = df[(df.departement == 75) & (df.date_transaction.str.startswith('2022-'))].copy()

    # Ajout de nouvelles colonnes pour la somme de chaque colonne de surface
    # ... (code pour le traitement des colonnes de surface)

    # Filtrage des lignes où la somme de toutes les colonnes de surface est égale à 0
    paris_df = paris_df[paris_df[[c + '_sum' for c in surface_cols]].sum(axis=1) == 0]

    # ... (code pour le reste du traitement)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

    # Entraînement des modèles
    train

# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings("ignore")

# Chargement du dataset
csv_file_path = 'transactions.csv'
df = pd.read_csv(csv_file_path)

# Suppression d'une colonne inutile
df = df.drop('Unnamed: 0', axis=1)

# Création d'une nouvelle fonction 'prix_m2'
df['prix_m2'] = df['prix'] / df['surface_habitable']

# Filtrage des données pour les transactions à Paris en 2022
selected_departments = [75, 77, 78, 91, 92, 94, 95]
paris_df = df[(df.departement.isin(selected_departments)) & (df.date_transaction.str.startswith('2022-'))].copy()

# Processus des colonnes de surface
surface_cols = [c for c in paris_df.columns if 'surface_' in c and c != 'surface_habitable']
for c in surface_cols:
    paris_df[c + '_sum'] = paris_df[c].apply(lambda x: sum(eval(x)) if 'NULL' not in x else 0)

# Filtrage des lignes où la somme des colonnes de surface est égale à 0
paris_df = paris_df[paris_df[[c + '_sum' for c in surface_cols]].sum(axis=1) == 0]

# Conversion de 'date_transaction' en datetime et calcul des jours depuis l'époque
paris_df['date_transaction'] = pd.to_datetime(paris_df['date_transaction'])
reference_date = pd.to_datetime('1970-01-01')
paris_df['days_since_epoch'] = (paris_df['date_transaction'] - reference_date).dt.days

# Suppression de la colonne originale 'date_transaction'
paris_df = paris_df.drop('date_transaction', axis=1)

# Création de variables dummy pour 'type_batiment'
paris_df = pd.get_dummies(paris_df, columns=['type_batiment'], prefix='type')

# Sélection des caractéristiques pertinentes pour X et y
X = paris_df[['days_since_epoch', 'longitude', 'latitude', 'id_ville', 'type_Appartement', 'type_Maison', 'vefa', 'n_pieces', 'surface_habitable']]
y = paris_df['prix_m2']

# Séparation des colonnes catégorielles et numériques
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Sauvegarde du DataFrame traité dans un nouveau fichier
dataset_file_path = 'dataset.csv'
paris_df.to_csv(dataset_file_path, index=False)

# Affichage du chemin du fichier du dataset
print("Chemin du fichier du dataset :", dataset_file_path)

# Création des transformateurs pour les données numériques et catégorielles
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Création d'un transformateur de colonnes pour appliquer les transformations aux colonnes appropriées
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Création d'un pipeline pour prétraiter les données
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Prétraitement des caractéristiques
X_processed = model_pipeline.fit_transform(X)
# Transformation de la colonne 'type_batiment' en colonnes booléennes
X_processed = pd.get_dummies(X_processed, columns=['type_batiment'], drop_first=True)

# Division du dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.8, random_state=42)

# Grille des paramètres pour la recherche
params_grid = {
    'DTR': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': range(1, 101, 10),
            'min_samples_split': range(2, 21, 2)
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': range(1, 51, 5)
        }
    },
    'LR': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
    },
    'RFR': {
        'model': RandomForestRegressor(),
        'params': {
            'max_depth': range(10, 111, 10),
            'min_samples_leaf': range(1, 12, 2),
            'n_estimators': [100, 200, 300, 400, 500]
        }
    }
}

# Entraînement des modèles avec validation croisée
best_models = {}  # Dictionnaire pour stocker les meilleurs modèles

for model_name, model_config in tqdm(params_grid.items()):
    gs = GridSearchCV(estimator=model_config
['model'], param_grid=model_config['params'])
gs.fit(X_train, y_train)
best_model = gs.best_estimator_
best_params = gs.best_params_
train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
score = best_model.score(X_test, y_test)
# Enregistrement du meilleur modèle
best_models[model_name] = {
    'model': best_model,
    'params': best_params,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'score': score
}

# Affichage des résultats
print(f"Model: {model_name}")
print(f"Optimal params: {best_params}")
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Model Score: {score}")
print()
#Entraînement automatique du meilleur modèle trouvé
best_model_name = max(best_models, key=lambda k: best_models[k]['score'])
best_model_to_train = best_models[best_model_name]['model']

print(f"\nEntraînement automatique du meilleur modèle trouvé ({best_model_name}) avec les meilleurs paramètres...")
best_model_to_train.fit(X_processed, y) # Utilisation de l'ensemble complet pour l'entraînement
print("Entraînement terminé.")
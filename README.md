
# Script de Formation de Modèle pour la Prédiction des Prix Immobiliers à Paris

## 1. Description du script
Ce script est conçu pour prédire les prix immobiliers au mètre carré à Paris pour l'année 2022. Le processus implique plusieurs étapes clés :
- **Nettoyage et prétraitement des données** : traitement des données brutes pour les rendre adaptées à l'analyse.
- **Ingénierie des caractéristiques** : transformation et sélection des caractéristiques pour améliorer la performance du modèle.
- **Sélection et entraînement des modèles** : utilisation de différents algorithmes de régression pour trouver le meilleur modèle.
- **Optimisation des hyperparamètres** : réglage fin des modèles pour améliorer leur précision.
- **Évaluation des modèles** : utilisation de métriques pour évaluer et comparer les performances des modèles.

## 2. Guide d'utilisation du script
Pour utiliser ce script, suivez ces étapes :
1. **Installation des dépendances** : Assurez-vous que les bibliothèques nécessaires (pandas, sklearn, matplotlib, numpy, pickle, tqdm) sont installées.
2. **Exécution du script** : Lancez le script dans un environnement Python. Le script lira le fichier de données, effectuera le prétraitement, l'entraînement des modèles, et sauvegardera le meilleur modèle.
3. **Résultats** : Le script affiche les performances de chaque modèle et sauvegarde le meilleur modèle pour une utilisation future.

## 3. Explication détaillée pour un débutant
### Nettoyage et prétraitement des données
```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

csv_file_path = 'transactions.csv'
df = pd.read_csv(csv_file_path)
...
```
- Importation de la bibliothèque `pandas` pour la manipulation des données et suppression des avertissements pour une sortie plus propre.
- Lecture des données à partir d'un fichier CSV.

### Calcul du prix au mètre carré
```python
df['prix_m2'] = df['prix'] / df['surface_habitable']
...
```
- Création d'une nouvelle colonne `prix_m2` en divisant le prix par la surface habitable.

### Séparation du jeu de données en entraînement et test
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
...
```
- Utilisation de `train_test_split` pour diviser les données en ensembles d'entraînement et de test.

### Entraînement des modèles avec validation croisée
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
...
for model_name, model_config in tqdm(params_grid.items()):
    ...
    gs = GridSearchCV(estimator=model_config['model'], param_grid=model_config['params'], verbose=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    ...
```
- Utilisation de `GridSearchCV` pour l'optimisation des hyperparamètres.
- Entraînement de chaque modèle et sélection du meilleur en fonction de la performance.

### Sauvegarde du meilleur modèle
```python
import pickle
pickle_file_path = 'best_model.pkl'
with open(pickle_file_path, 'wb') as file:
    pickle.dump(best_model_to_train, file)
...
```
- Utilisation de `pickle` pour sauvegarder le meilleur modèle pour une utilisation future.

Ce script fournit un cadre pour la prédiction des prix immobiliers à Paris, et peut être adapté pour d'autres tâches de régression similaires.

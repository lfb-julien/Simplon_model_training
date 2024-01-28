from fastapi import FastAPI
import pickle
import uvicorn
app = FastAPI()

@app.get("/predict")
def predict(longitude: float, latitude: float):
    # Charger le modèle à partir du fichier best_model.pkl
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Effectuer la prédiction avec le modèle chargé
    prediction = model.predict([[longitude, latitude]])

    # Retourner le résultat de la prédiction
    return {"prediction": prediction}

# Exécuter l'application avec uvicorn (uniquement si ce n'est pas chargé en tant que module)
if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1")

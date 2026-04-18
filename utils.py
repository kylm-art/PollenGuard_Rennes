import os
import requests
import pandas as pd


def charger_donnees_api(
    url: str,
    params: dict,
    fichier_cache: str,
    colonne_date: str = "time",
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Charge des données depuis une API (Open-Meteo ou autre) avec gestion du cache.

    Paramètres
    ----------
    url : str
        URL de l'API à interroger.
    params : dict
        Paramètres de la requête API.
    fichier_cache : str
        Chemin du fichier CSV pour stocker les données localement.
    colonne_date : str, optionnel
        Nom de la colonne contenant les dates dans la réponse JSON.
    force_reload : bool, optionnel
        Si True, force le rechargement des données via API.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant les données récupérées.
    """

    # Cache
    if os.path.exists(fichier_cache) and not force_reload:
        return pd.read_csv(fichier_cache, parse_dates=["date"])

    # Requête API
    response = requests.get(url, params=params)
    data = response.json()

    # Construction DataFrame
    df = pd.DataFrame(data["hourly"])

    # Harmonisation
    df = df.rename(columns={colonne_date: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Sauvegarde
    df.to_csv(fichier_cache, index=False)

    return df

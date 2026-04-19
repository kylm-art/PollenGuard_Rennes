import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def identifier_plages_manquantes(df: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """
    Identifie les dates de début et de fin des périodes continues où les données sont manquantes,
    en calculant le total cumulé pour vérification.

    Parameters
    ----------
    df : pd.DataFrame
        Le jeu de données contenant une colonne 'date'.
    colonne : str
        Le nom de la colonne à analyser (ex: 'pollen_bouleau').

    Returns
    -------
    pd.DataFrame
        Un tableau résumant chaque période manquante avec le total cumulé.
    """
    # Création d'un masque booléen (True si la donnée est manquante)
    est_manquant = df[colonne].isna()
    total_nan_reel = est_manquant.sum()

    if total_nan_reel == 0:
        print(f"Information : Aucune valeur manquante détectée pour '{colonne}'.")
        return pd.DataFrame()

    #  Identification des blocs continus
    blocs = (est_manquant != est_manquant.shift()).cumsum()

    #  Isolation et agrégation
    df_manquant = df[est_manquant].copy()
    df_manquant['id_bloc'] = blocs[est_manquant]

    plages = df_manquant.groupby('id_bloc')['date'].agg(
        date_debut='min',
        date_fin='max',
        heures_manquantes='count'
    ).reset_index(drop=True)

    #  Ajout du total cumulé et vérification
    plages['total_cumule_manquant'] = plages['heures_manquantes'].cumsum()

    total_identifie = plages['heures_manquantes'].sum()

    print(f"--- Analyse : {colonne} ---")
    print(f"Total NaN dans la colonne : {total_nan_reel}")
    print(f"Total identifié par blocs : {total_identifie}")

    if total_nan_reel == total_identifie:
        print("Vérification : OK (Tous les NaN sont comptabilisés).")
    else:
        print("Attention : Écart détecté dans le comptage.")

    return plages


def imputer_na_par_valeur(df: pd.DataFrame, colonnes: list, valeur: float = 0.0) -> pd.DataFrame:
    """
    Impute une valeur spécifique (par défaut 0) dans les colonnes choisies.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame à traiter.
    colonnes : list
        La liste des noms de colonnes à imputer (ex: ['pollen_bouleau']).
    valeur : float, optional (default=0.0)
        La valeur de remplacement.

    Returns
    -------
    pd.DataFrame
        Le DataFrame avec les valeurs imputées.
    """
    df_impute = df.copy()
    for col in colonnes:
        if col in df_impute.columns:
            df_impute[col] = df_impute[col].fillna(valeur)
        else:
            print(f"Avertissement : La colonne '{col}' n'existe pas dans le DataFrame.")

    return df_impute


def tracer_series_temporelles(
    df,
    variables: list,
    titres: dict = None,
    ylabels: dict = None,
    couleurs: dict = None
):
    """
    Trace plusieurs séries temporelles à partir d'une liste de variables.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données (avec une colonne 'date').
    variables : list
        Liste des variables à tracer.
    titres : dict, optionnel
        Dictionnaire {variable: titre du graphique}.
    ylabels : dict, optionnel
        Dictionnaire {variable: label de l'axe Y}.
    couleurs : dict, optionnel
        Dictionnaire {variable: couleur}.
    """

    sns.set_style("ticks")

    for var in variables:
        plt.figure(figsize=(15, 5))

        # Valeurs par défaut si non fournies
        titre = titres[var] if titres and var in titres else f"Évolution de {var}"
        ylabel = ylabels[var] if ylabels and var in ylabels else var.replace("_", " ").capitalize()
        couleur = couleurs[var] if couleurs and var in couleurs else "#1f77b4"

        # Tracé
        sns.lineplot(data=df, x="date", y=var, color=couleur)

        # Mise en forme
        plt.title(titre, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Temps", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


def sauvegarder_donnees_clean(
    df: pd.DataFrame,
    nom_fichier: str = "donnees_clean.csv",
    dossier: str = "data/clean"
) -> None:
    """
    Sauvegarde un DataFrame dans un dossier donné.
    Crée le dossier s'il n'existe pas.

    Paramètres
    ----------
    df : pd.DataFrame
        Données à sauvegarder.
    nom_fichier : str, optionnel
        Nom du fichier de sortie.
    dossier : str, optionnel
        Chemin du dossier de sauvegarde.

    Retour
    ------
    None
    """

    # Création du dossier si inexistant
    os.makedirs(dossier, exist_ok=True)

    # Chemin complet du fichier
    chemin_fichier = os.path.join(dossier, nom_fichier)

    # Sauvegarde
    df.to_csv(chemin_fichier, index=False)

    print(f"Données sauvegardées dans : {chemin_fichier}")


def classifier_risque(valeur):
    """Classifie le niveau de risque allergique selon les seuils RNSA.

    Paramètres
    ----------
    valeur : float
        Concentration de pollen en grains/m³

    Retourne
    -------
    int : 0 (Faible ≤30), 1 (Modéré 31-80), 2 (Élevé >80)
    """
    if valeur <= 30:
        return 0
    elif valeur <= 80:
        return 1
    else:
        return 2

def construire_features(df, colonnes_pollen, date_col='date'):
    """
    Construit les variables explicatives pour la modélisation.
    
    Paramètres
    ----------
    df : DataFrame source
    colonnes_pollen : list, noms des colonnes de pollen
    date_col : str, nom de la colonne date
         
    Retourne
    -------
    DataFrame enrichi avec variables temporelles et retardées
    """
    df = df.copy()
    
    # Variables temporelles
    df['mois'] = df[date_col].dt.month
    df['jour_annee'] = df[date_col].dt.dayofyear
    
    # Variables retardées et moyenne glissante pour chaque colonne pollen
    for col in colonnes_pollen:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df[f'{col}_moy3j'] = df[col].rolling(3).mean()
    
    df = df.dropna()
    return df
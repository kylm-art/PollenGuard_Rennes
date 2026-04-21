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

def creer_cible_binaire(df, colonnes_pollen):
    """
    Crée une variable cible binaire pour chaque pollen.
    
    Paramètres
    ----------
    df : DataFrame source
    colonnes_pollen : list, noms des colonnes de pollen
    
    Retourne
    -------
    DataFrame avec nouvelles colonnes binaires (0=Faible, 1=À risque)
    """
    df = df.copy()
    for col in colonnes_pollen:
        df[f'risque_bin_{col}_j1'] = (df[col] > 30).astype(int).shift(-1)
    df = df.dropna()
    for col in colonnes_pollen:
        df[f'risque_bin_{col}_j1'] = df[f'risque_bin_{col}_j1'].astype(int)
    return df    


def predire_risque(date_cible, modele_bouleau, modele_graminees, features):
    """
    Prédit le niveau de risque allergique pour une date donnée.
    
    Paramètres
    ----------
    date_cible : str, format 'YYYY-MM-DD'
    modele_bouleau : modèle chargé avec joblib
    modele_graminees : modèle chargé avec joblib
    features : list, noms des features dans le bon ordre
    
    Retourne
    -------
    dict avec les prédictions pour bouleau et graminées
    """
    import requests
    import pandas as pd
    import numpy as np
    from datetime import timedelta

    date_cible = pd.Timestamp(date_cible)
    date_debut = (date_cible - timedelta(days=20)).strftime('%Y-%m-%d')
    date_fin = date_cible.strftime('%Y-%m-%d')

    # Récupération météo
    url_meteo = "https://archive-api.open-meteo.com/v1/archive"
    params_meteo = {
        "latitude": 48.11, "longitude": -1.67,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
        "start_date": date_debut, "end_date": date_fin,
        "timezone": "Europe/Paris"
    }
    r_meteo = requests.get(url_meteo, params=params_meteo).json()
    df_meteo = pd.DataFrame({
        "date": pd.to_datetime(r_meteo["hourly"]["time"]),
        "temperature": r_meteo["hourly"]["temperature_2m"],
        "precipitations": r_meteo["hourly"]["precipitation"],
        "vitesse_vent": r_meteo["hourly"]["wind_speed_10m"]
    })

    # Récupération pollen
    url_pollen = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_pollen = {
        "latitude": 48.11, "longitude": -1.67,
        "hourly": ["birch_pollen", "grass_pollen"],
        "start_date": date_debut, "end_date": date_fin
    }
    r_pollen = requests.get(url_pollen, params=params_pollen).json()
    df_pollen = pd.DataFrame({
        "date": pd.to_datetime(r_pollen["hourly"]["time"]),
        "pollen_bouleau": r_pollen["hourly"]["birch_pollen"],
        "pollen_graminees": r_pollen["hourly"]["grass_pollen"]
    })

    # Fusion et agrégation journalière
    df = pd.merge(df_pollen, df_meteo, on="date", how="inner")
    df_jour = df.resample('D', on='date').agg(
        pollen_bouleau=('pollen_bouleau', 'max'),
        pollen_graminees=('pollen_graminees', 'max'),
        temperature=('temperature', 'mean'),
        precipitations=('precipitations', 'sum'),
        vitesse_vent=('vitesse_vent', 'mean')
    ).reset_index()

    # Imputation NaN
    df_jour[['pollen_bouleau', 'pollen_graminees']] = \
        df_jour[['pollen_bouleau', 'pollen_graminees']].fillna(0)

    # Construction des features
    df_jour['jour_de_annee'] = df_jour['date'].dt.dayofyear
    df_jour['mois'] = df_jour['date'].dt.month
    df_jour['annee'] = df_jour['date'].dt.year
    df_jour['saison'] = df_jour['mois'].map({
        12:0,1:0,2:0, 3:1,4:1,5:1,
        6:2,7:2,8:2, 9:3,10:3,11:3
    })
    df_jour['jour_sin'] = np.sin(2*np.pi*df_jour['jour_de_annee']/365)
    df_jour['jour_cos'] = np.cos(2*np.pi*df_jour['jour_de_annee']/365)
    df_jour['mois_sin'] = np.sin(2*np.pi*df_jour['mois']/12)
    df_jour['mois_cos'] = np.cos(2*np.pi*df_jour['mois']/12)

    df_jour['temp_lag1']  = df_jour['temperature'].shift(1)
    df_jour['temp_roll7'] = df_jour['temperature'].shift(1).rolling(7).mean()
    df_jour['precip_lag1']  = df_jour['precipitations'].shift(1)
    df_jour['precip_lag2']  = df_jour['precipitations'].shift(2)
    df_jour['precip_roll7'] = df_jour['precipitations'].shift(1).rolling(7).sum()
    df_jour['vitesse_vent_lag1']  = df_jour['vitesse_vent'].shift(1)
    df_jour['vitesse_vent_roll7'] = df_jour['vitesse_vent'].shift(1).rolling(7).mean()

    df_jour['gdd_daily'] = (df_jour['temperature'] - 5).clip(lower=0)
    df_jour['gdd_cumul'] = df_jour['gdd_daily'].cumsum()

    for col in ['pollen_bouleau', 'pollen_graminees']:
        for lag in [1, 2, 3]:
            df_jour[f'{col}_lag{lag}'] = df_jour[col].shift(lag)
        df_jour[f'{col}_moy3j'] = df_jour[col].shift(1).rolling(3).mean()

    # Prendre la dernière ligne = jour cible
    derniere_ligne = df_jour.dropna().iloc[[-1]]
    X = derniere_ligne[features]

    # Prédiction
    pred_bouleau = modele_bouleau.predict(X)[0]
    pred_graminees = modele_graminees.predict(X)[0]

    labels = {0: 'Faible', 1: 'À risque'}

    return {
        "date": date_cible.strftime('%Y-%m-%d'),
        "bouleau": labels[pred_bouleau],
        "graminees": labels[pred_graminees]
    }


def _construire_df_features(df_jour):
    """Fonction interne partagée : construit toutes les features à partir d'un df journalier."""
    df = df_jour.copy()
    df["jour_de_annee"] = df["date"].dt.dayofyear
    df["mois"]          = df["date"].dt.month
    df["annee"]         = df["date"].dt.year
    df["saison"]        = df["mois"].map({
        12:0,1:0,2:0, 3:1,4:1,5:1,
        6:2,7:2,8:2,  9:3,10:3,11:3
    })
    df["jour_sin"] = np.sin(2*np.pi*df["jour_de_annee"]/365)
    df["jour_cos"] = np.cos(2*np.pi*df["jour_de_annee"]/365)
    df["mois_sin"] = np.sin(2*np.pi*df["mois"]/12)
    df["mois_cos"] = np.cos(2*np.pi*df["mois"]/12)
    df["temp_lag1"]          = df["temperature"].shift(1)
    df["temp_roll7"]         = df["temperature"].shift(1).rolling(7).mean()
    df["precip_lag1"]        = df["precipitations"].shift(1)
    df["precip_lag2"]        = df["precipitations"].shift(2)
    df["precip_roll7"]       = df["precipitations"].shift(1).rolling(7).sum()
    df["vitesse_vent_lag1"]  = df["vitesse_vent"].shift(1)
    df["vitesse_vent_roll7"] = df["vitesse_vent"].shift(1).rolling(7).mean()
    df["gdd_daily"]          = (df["temperature"] - 5).clip(lower=0)
    df["gdd_cumul"]          = df["gdd_daily"].cumsum()
    for col in ["pollen_bouleau", "pollen_graminees"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_moy3j"] = df[col].shift(1).rolling(3).mean()
    return df


def _fusionner_agreger(df_pollen, df_meteo):
    """Fusionne pollen + météo horaires et agrège en journalier."""
    df = pd.merge(df_pollen, df_meteo, on="date", how="inner")
    df_jour = df.resample("D", on="date").agg(
        pollen_bouleau=("pollen_bouleau",   "max"),
        pollen_graminees=("pollen_graminees","max"),
        temperature=("temperature",         "mean"),
        precipitations=("precipitations",   "sum"),
        vitesse_vent=("vitesse_vent",        "mean"),
    ).reset_index()
    df_jour[["pollen_bouleau","pollen_graminees"]] = \
        df_jour[["pollen_bouleau","pollen_graminees"]].fillna(0)
    return df_jour

def charger_meteo_recente(jours: int = 120) -> pd.DataFrame:
    """
    Charge météo + pollen des derniers N jours depuis l'API (agrégation journalière).
    jours=120 permet de couvrir depuis le 1er janvier (pour le GDD cumulé annuel).
    Retourne : date, pollen_bouleau, pollen_graminees, temperature,
               precipitations, vitesse_vent
    """
    from datetime import datetime, timedelta
    aujourd_hui = datetime.today()
    date_debut  = (aujourd_hui - timedelta(days=jours)).strftime("%Y-%m-%d")
    date_fin    = aujourd_hui.strftime("%Y-%m-%d")

    r_meteo = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude":48.11,"longitude":-1.67,
                "hourly":["temperature_2m","precipitation","wind_speed_10m"],
                "start_date":date_debut,"end_date":date_fin,"timezone":"Europe/Paris"},
        timeout=30,
    ).json()
    df_meteo = pd.DataFrame({
        "date":          pd.to_datetime(r_meteo["hourly"]["time"]),
        "temperature":   r_meteo["hourly"]["temperature_2m"],
        "precipitations":r_meteo["hourly"]["precipitation"],
        "vitesse_vent":  r_meteo["hourly"]["wind_speed_10m"],
    })

    r_pollen = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={"latitude":48.11,"longitude":-1.67,
                "hourly":["birch_pollen","grass_pollen"],
                "start_date":date_debut,"end_date":date_fin},
        timeout=30,
    ).json()
    df_pollen = pd.DataFrame({
        "date":             pd.to_datetime(r_pollen["hourly"]["time"]),
        "pollen_bouleau":   r_pollen["hourly"]["birch_pollen"],
        "pollen_graminees": r_pollen["hourly"]["grass_pollen"],
    })

    return _fusionner_agreger(df_pollen, df_meteo)


def predire_risque_complet(date_cible, modele_bouleau, modele_graminees,
                            features, seuil: float = 0.5) -> dict:
    """
    Appelle predire_risque (qui fonctionne), puis tente d'appliquer
    le seuil de sensibilité via predict_proba si disponible.
    """
    # On utilise predire_risque qui fonctionne de façon fiable
    resultat_base = predire_risque(date_cible, modele_bouleau,
                                   modele_graminees, features)

    # Tentative d'ajout des probabilités pour le seuil de sensibilité
    try:
        from datetime import timedelta
        date_cible_ts = pd.Timestamp(date_cible)
        date_debut    = (date_cible_ts - timedelta(days=20)).strftime("%Y-%m-%d")
        date_fin      = date_cible_ts.strftime("%Y-%m-%d")

        r_meteo = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude":48.11,"longitude":-1.67,
                    "hourly":["temperature_2m","precipitation","wind_speed_10m"],
                    "start_date":date_debut,"end_date":date_fin,
                    "timezone":"Europe/Paris"}
        ).json()
        df_meteo = pd.DataFrame({
            "date":           pd.to_datetime(r_meteo["hourly"]["time"]),
            "temperature":    r_meteo["hourly"]["temperature_2m"],
            "precipitations": r_meteo["hourly"]["precipitation"],
            "vitesse_vent":   r_meteo["hourly"]["wind_speed_10m"],
        })

        r_pollen = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude":48.11,"longitude":-1.67,
                    "hourly":["birch_pollen","grass_pollen"],
                    "start_date":date_debut,"end_date":date_fin}
        ).json()
        df_pollen = pd.DataFrame({
            "date":             pd.to_datetime(r_pollen["hourly"]["time"]),
            "pollen_bouleau":   r_pollen["hourly"]["birch_pollen"],
            "pollen_graminees": r_pollen["hourly"]["grass_pollen"],
        })

        df_jour = _fusionner_agreger(df_pollen, df_meteo)
        df_feat = _construire_df_features(df_jour).dropna()
        X = df_feat.iloc[[-1]][features]

        proba_b = float(modele_bouleau.predict_proba(X)[0][1])
        proba_g = float(modele_graminees.predict_proba(X)[0][1])

        # Applique le seuil de sensibilité
        labels = {True: "À risque", False: "Faible"}
        return {
            "date":            resultat_base["date"],
            "bouleau":         labels[proba_b >= seuil],
            "graminees":       labels[proba_g >= seuil],
            "proba_bouleau":   proba_b,
            "proba_graminees": proba_g,
            "source":          "api",
        }

    except Exception:
        # Si les probas échouent, on retourne le résultat de base sans seuil
        return {
            **resultat_base,
            "proba_bouleau":   None,
            "proba_graminees": None,
            "source":          "api",
        }

def predire_risque_futur(jours_avance: int, modele_bouleau, modele_graminees,
                          features, seuil: float = 0.5) -> dict:
    """
    Prédit le risque pour J+1, J+2 ou J+3 en combinant :
    - API archive (20 derniers jours) pour calculer les lags historiques
    - API forecast (Open-Meteo + Air Quality) pour les données futures

    Fiabilité décroissante : J+1 > J+2 > J+3
    """
    from datetime import datetime, timedelta
    aujourd_hui = datetime.today()
    date_cible  = aujourd_hui + timedelta(days=jours_avance)
    date_arch_debut  = (aujourd_hui - timedelta(days=25)).strftime("%Y-%m-%d")
    date_arch_fin    = aujourd_hui.strftime("%Y-%m-%d")
    date_fore_debut  = (aujourd_hui + timedelta(days=1)).strftime("%Y-%m-%d")
    date_fore_fin    = date_cible.strftime("%Y-%m-%d")

    # ── Archive météo ─────────────────────────────────────────────
    r_ma = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude":48.11,"longitude":-1.67,
                "hourly":["temperature_2m","precipitation","wind_speed_10m"],
                "start_date":date_arch_debut,"end_date":date_arch_fin,
                "timezone":"Europe/Paris"}
    ).json()
    df_ma = pd.DataFrame({
        "date":          pd.to_datetime(r_ma["hourly"]["time"]),
        "temperature":   r_ma["hourly"]["temperature_2m"],
        "precipitations":r_ma["hourly"]["precipitation"],
        "vitesse_vent":  r_ma["hourly"]["wind_speed_10m"],
    })

    # ── Forecast météo ────────────────────────────────────────────
    r_mf = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude":48.11,"longitude":-1.67,
                "hourly":["temperature_2m","precipitation","wind_speed_10m"],
                "start_date":date_fore_debut,"end_date":date_fore_fin,
                "timezone":"Europe/Paris"}
    ).json()
    df_mf = pd.DataFrame({
        "date":          pd.to_datetime(r_mf["hourly"]["time"]),
        "temperature":   r_mf["hourly"]["temperature_2m"],
        "precipitations":r_mf["hourly"]["precipitation"],
        "vitesse_vent":  r_mf["hourly"]["wind_speed_10m"],
    })

    # ── Archive pollen ────────────────────────────────────────────
    url_aq = "https://air-quality-api.open-meteo.com/v1/air-quality"
    r_pa = requests.get(url_aq, params={
        "latitude":48.11,"longitude":-1.67,
        "hourly":["birch_pollen","grass_pollen"],
        "start_date":date_arch_debut,"end_date":date_arch_fin,
    }).json()
    df_pa = pd.DataFrame({
        "date":             pd.to_datetime(r_pa["hourly"]["time"]),
        "pollen_bouleau":   r_pa["hourly"]["birch_pollen"],
        "pollen_graminees": r_pa["hourly"]["grass_pollen"],
    })

    # ── Forecast pollen ───────────────────────────────────────────
    r_pf = requests.get(url_aq, params={
        "latitude":48.11,"longitude":-1.67,
        "hourly":["birch_pollen","grass_pollen"],
        "start_date":date_fore_debut,"end_date":date_fore_fin,
    }).json()
    df_pf = pd.DataFrame({
        "date":             pd.to_datetime(r_pf["hourly"]["time"]),
        "pollen_bouleau":   r_pf["hourly"]["birch_pollen"],
        "pollen_graminees": r_pf["hourly"]["grass_pollen"],
    })

    # ── Fusion + features 
    df_arch = _fusionner_agreger(df_pa, df_ma)
    df_fore = _fusionner_agreger(df_pf, df_mf)
    df_jour = pd.concat([df_arch, df_fore], ignore_index=True)
    df_feat = _construire_df_features(df_jour).dropna()

    row = df_feat[df_feat["date"].dt.date == date_cible.date()]
    if row.empty:
        row = df_feat.iloc[[-1]]

    X = row[features]

    try:
        proba_b = float(modele_bouleau.predict_proba(X)[0][1])
        proba_g = float(modele_graminees.predict_proba(X)[0][1])
        pred_b  = "À risque" if proba_b >= seuil else "Faible"
        pred_g  = "À risque" if proba_g >= seuil else "Faible"
    except Exception:
        proba_b = proba_g = None
        labels  = {0:"Faible", 1:"À risque"}
        pred_b  = labels[int(modele_bouleau.predict(X)[0])]
        pred_g  = labels[int(modele_graminees.predict(X)[0])]

    return {
        "date":            date_cible.strftime("%Y-%m-%d"),
        "bouleau":         pred_b,
        "graminees":       pred_g,
        "proba_bouleau":   proba_b,
        "proba_graminees": proba_g,
        "source":          "forecast",
    }


def predire_depuis_s3(date_cible, modele_b, modele_g,
                       features, df_s3: pd.DataFrame) -> dict:
    """Fallback S3 : features déjà précalculées dans data_features_final_clean.csv."""
    ts  = pd.Timestamp(date_cible)
    row = df_s3[df_s3["date"].dt.date == ts.date()]
    if row.empty:
        raise ValueError(f"Date {date_cible} absente du cache S3.")
    X      = row[features]
    labels = {0:"Faible", 1:"À risque"}
    return {
        "date":      str(date_cible),
        "bouleau":   labels[int(modele_b.predict(X)[0])],
        "graminees": labels[int(modele_g.predict(X)[0])],
        "proba_bouleau":   None,
        "proba_graminees": None,
        "source":    "cache",
    }


def charger_s3_modele() -> pd.DataFrame:
    """Charge data_features_final_clean.csv depuis S3 (fallback prédiction)."""
    try:
        import s3fs
        fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url":"https://minio.lab.sspcloud.fr"}
        )
        with fs.open("lesline/diffusion/data_features_final_clean.csv") as f:
            return pd.read_csv(f, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()


def charger_historique_pollen() -> pd.DataFrame:
    """
    Charge pollen 2021–2024 depuis Open-Meteo Air Quality API.
    Fallback sur S3 si l'API échoue.
    Retourne : date, pollen_bouleau, pollen_graminees, annee, jour_de_annee
    """
    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude":48.11,"longitude":-1.67,
                    "hourly":["birch_pollen","grass_pollen"],
                    "start_date":"2021-01-01","end_date":"2026-04-19"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame({
            "date":             pd.to_datetime(data["hourly"]["time"]),
            "pollen_bouleau":   data["hourly"]["birch_pollen"],
            "pollen_graminees": data["hourly"]["grass_pollen"],
        })
        df = df.resample("D", on="date").agg(
            pollen_bouleau=("pollen_bouleau",   "max"),
            pollen_graminees=("pollen_graminees","max"),
        ).reset_index()
        df[["pollen_bouleau","pollen_graminees"]] = \
            df[["pollen_bouleau","pollen_graminees"]].fillna(0)
        df["annee"]         = df["date"].dt.year
        df["jour_de_annee"] = df["date"].dt.dayofyear
        return df
    except Exception:
        try:
            import s3fs
            fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url":"https://minio.lab.sspcloud.fr"}
            )
            with fs.open("lesline/diffusion/data_pollen_meteo_jour.csv") as f:
                df = pd.read_csv(f, parse_dates=["date"])
            df["annee"]         = df["date"].dt.year
            df["jour_de_annee"] = df["date"].dt.dayofyear
            return df
        except Exception:
            return pd.DataFrame()


def charger_donnees_pour_date(date_cible, jours: int = 120) -> pd.DataFrame:
    """
    Charge météo + pollen des 'jours' jours précédant date_cible.
    Utilisé pour calculer les KPIs pour n'importe quelle date passée.
    Retourne un DataFrame journalier avec :
        date, pollen_bouleau, pollen_graminees,
        temperature, precipitations, vitesse_vent
    """
    date_cible = pd.Timestamp(date_cible)
    date_fin   = date_cible.strftime("%Y-%m-%d")
    date_debut = (date_cible - pd.Timedelta(days=jours)).strftime("%Y-%m-%d")

    r_meteo = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 48.11, "longitude": -1.67,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_10m"],
            "start_date": date_debut, "end_date": date_fin,
            "timezone": "Europe/Paris"
        },
        timeout=30,
    ).json()
    df_meteo = pd.DataFrame({
        "date":           pd.to_datetime(r_meteo["hourly"]["time"]),
        "temperature":    r_meteo["hourly"]["temperature_2m"],
        "precipitations": r_meteo["hourly"]["precipitation"],
        "vitesse_vent":   r_meteo["hourly"]["wind_speed_10m"],
    })

    r_pollen = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": 48.11, "longitude": -1.67,
            "hourly": ["birch_pollen", "grass_pollen"],
            "start_date": date_debut, "end_date": date_fin,
        },
        timeout=30,
    ).json()
    df_pollen = pd.DataFrame({
        "date":             pd.to_datetime(r_pollen["hourly"]["time"]),
        "pollen_bouleau":   r_pollen["hourly"]["birch_pollen"],
        "pollen_graminees": r_pollen["hourly"]["grass_pollen"],
    })

    return _fusionner_agreger(df_pollen, df_meteo)
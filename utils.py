import os
import requests
import pandas as pd


def load_data_url(
    url,
    data_dir,
    file_name,
    header_row=0,
    force_reload=False,
    sep=None
) -> pd.DataFrame:
    """
    Charge un fichier de données depuis une URL (CSV ou Excel) et le stocke en local.

    Parameters
    ----------
    url : str
        URL du fichier source (CSV ou Excel).
    data_dir : str
        Dossier de stockage local.
    file_name : str
        Nom du fichier CSV local.
    header_row : int
        Ligne contenant les noms de colonnes (0-indexé, utilisé pour Excel).
    force_reload : bool
        Si True, recharge depuis la source même si le fichier existe.
    sep : str or None
        Séparateur pour les fichiers CSV. Si None, détection automatique.

    Returns
    -------
    pd.DataFrame
        Données chargées sous forme de DataFrame.
    """

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path) and not force_reload:
        return pd.read_csv(file_path)

    # Détection du type de fichier
    if url.endswith(".xlsx") or url.endswith(".xls"):
        df = pd.read_excel(url, header=header_row)
    else:
        df = pd.read_csv(url, sep=sep if sep else None, engine="python")

    # Sauvegarde en CSV standard (séparateur virgule)
    df.to_csv(file_path, index=False)

    return df


def load_bpe(
    base_url,
    limit,
    data_dir,
    file_name,
    force_reload=False
) -> pd.DataFrame:
    """
    Charge la BPE via API et la stocke en CSV sans transformation.

    Parameters
    ----------
    base_url : str
        URL de l'API BPE.
    limit : int
        Nombre de lignes par requête.
    data_dir : str
        Dossier de stockage local.
    file_name : str
        Nom du fichier CSV local.
    force_reload : bool
        Si True, recharge depuis l'API.

    Returns
    -------
    pd.DataFrame
        Données BPE brutes.
    """

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path) and not force_reload:
        return pd.read_csv(file_path)

    all_data = []
    offset = 0

    while True:
        url = f"{base_url}?limit={limit}&offset={offset}"
        response = requests.get(url)
        data = response.json()

        results = data.get("results", [])
        if not results:
            break

        all_data.extend(results)
        offset += limit

    df = pd.json_normalize(all_data)

    # Sauvegarde brute
    df.to_csv(file_path, index=False)

    return df

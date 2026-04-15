import os
import requests
import pandas as pd


def load_filosofi(
    url: str,
    header_row: int,
    data_dir: str,
    file_name: str,
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Charge la base FILOSOFI et la stocke en CSV sans transformation.

    Parameters
    ----------
    url : str
        URL du fichier Excel source.
    header_row : int
        Ligne contenant les noms de colonnes (0-indexé).
    data_dir : str
        Dossier de stockage local.
    file_name : str
        Nom du fichier CSV local.
    force_reload : bool
        Si True, recharge depuis la source même si le fichier existe.

    Returns
    -------
    pd.DataFrame
        Données FILOSOFI brutes.
    """

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path) and not force_reload:
        return pd.read_csv(file_path)

    df = pd.read_excel(url, header=header_row)

    # Sauvegarde brute en CSV
    df.to_csv(file_path, index=False)

    return df


def load_bpe(
    base_url: str,
    limit: int,
    data_dir: str,
    file_name: str,
    force_reload: bool = False
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
# 🌿 PollenGuard — Prédiction du Risque Pollinique à Rennes

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-déployé-green)
![Licence](https://img.shields.io/badge/Licence-Académique-orange)

> Projet Data Science · 2ème année ENSAI · 2025–2026

---

Ce travail a été réalisé dans le cadre du cours de Python pour la Data Science dispensé en 2ème année à l'ENSAI de Rennes.

---

##  Application en ligne

L'application est déployée et accessible publiquement :

**[PollenGuard — Accéder à l'application](https://pollenguardrennes-28c5n7eaekhhumszz8jk6r.streamlit.app/)**

Elle permet de :
- Consulter le bulletin allergie du jour sélectionné en temps réel 
- Visualiser les indicateurs météo-polliniques (GDD, lessivage, anomalie saisonnière, décalage phénologique)
- Suivre l'évolution du pollen sur les 30 derniers jours
- Consulter les prédictions J+1 pour le bouleau et les graminées

![Aperçu de l'application](ressources/app_screenshot.png)

---

##  Contexte et problématique

En France, près de **30% de la population souffre d'allergie pollinique**, un chiffre en hausse constante avec le changement climatique. Les allergènes polliniques (bouleau, graminées) provoquent des rhinites, conjonctivites et crises d'asthme dont la sévérité est directement liée à la concentration de pollen dans l'atmosphère.

**Question de recherche :**
> Peut-on prédire le niveau de risque allergique journalier à Rennes à partir des conditions météorologiques et des concentrations polliniques récentes ?

**Réponse apportée :** Deux modèles de Machine Learning (un par type de pollen) permettent de classifier le risque du lendemain en deux niveaux (Faible / À risque), avec une application web accessible au grand public pour consulter les prévisions en temps réel.

---

##  Résultats clés

| Pollen | Modèle | F1-macro (test) | Rappel "À risque" |
|--------|--------|-----------------|-------------------|
| Bouleau | Random Forest | **0.85** | 0.97 |
| Graminées | Régression Logistique | **0.96** | 0.96 |

> La classification binaire (Faible / À risque) a été préférée à une classification à 3 classes (Faible / Modéré / Élevé) pour améliorer la détection des épisodes allergiques réels et réduire le surapprentissage.

---

##  Structure du projet

```
G4B_pollenguard_rennes/
│
├── app/
│   └── main.py                    # Application Streamlit
│
├── data/
│   ├── raw/                       # Données brutes téléchargées
│   └── clean/                     # Données nettoyées et enrichies
│
├── models/
│   ├── modele_bouleau.pkl         # Modèle Random Forest — bouleau
│   ├── modele_graminees.pkl       # Modèle Régression Logistique — graminées
│   └── features.pkl               # Liste ordonnée des features du modèle
│
├── ressources/                    # Images et ressources visuelles
├── Rapport_final.ipynb            # Rapport complet : EDA, modélisation, évaluation
├── utils.py                       # Fonctions partagées (import, prédiction...)
├── pyproject.toml                 # Dépendances gérées par uv
├── python_version                 # Version de python
├── uv.lock                      # Versions exactes des packages (ne pas modifier)
└── README.md
```

---

##  Méthodologie

### Sources de données

Toutes les données proviennent de l'**API Open-Meteo** :

| Source | Variables | Granularité |
|--------|-----------|-------------|
| [Open-Meteo Archive](https://archive-api.open-meteo.com) | Température, précipitations, vitesse du vent | Horaire → agrégé en journalier |
| [Open-Meteo Air Quality](https://air-quality-api.open-meteo.com) | Pollen bouleau (`birch_pollen`), pollen graminées (`grass_pollen`) | Horaire → agrégé en journalier |

**Couverture temporelle :** 2021-01-01 → 2026-04-19  
**Localisation :** Rennes (48.11°N, 1.67°W)

### Construction de la variable cible

La variable cible est définie à partir de la concentration pollinique du lendemain (J+1), selon les seuils du RNSA :

| Classe | Seuil | Label binaire |
|--------|-------|---------------|
| Faible | ≤ 30 grains/m³ | 0 — Faible |
| Modéré | 31–80 grains/m³ | 1 — À risque |
| Élevé | > 80 grains/m³ | 1 — À risque |

### Features utilisées (22 variables)

| Groupe | Variables |
|--------|-----------|
| Temporelles | `jour_sin`, `jour_cos`, `mois_sin`, `mois_cos`, `saison`, `annee` |
| Température | `temp_lag1`, `temp_roll7`, `gdd_cumul` |
| Précipitations | `precip_lag1`, `precip_lag2`, `precip_roll7` |
| Vent | `vitesse_vent_lag1`, `vitesse_vent_roll7` |
| Historique pollen | `pollen_bouleau_lag1/2/3`, `pollen_bouleau_moy3j`, `pollen_graminees_lag1/2/3`, `pollen_graminees_moy3j` |

> Les variables laggées (`_lag`) et les moyennes glissantes (`_roll`, `_moy`) capturent l'inertie biologique : la floraison réagit à l'accumulation de chaleur sur plusieurs jours, pas à la température instantanée.

### Modèles

| Pollen | Algorithme | Split |
|--------|-----------|-------|
| Bouleau | Random Forest Classifier | Entraînement 2021–2024 · Test 2025–2026 |
| Graminées | Régression Logistique | Entraînement 2021–2024 · Test 2025–2026 |

> Le split est **temporel** (pas aléatoire) pour simuler les conditions réelles de prévision :  on ne peut pas utiliser le futur pour entraîner un modèle de prévision.

---

##  Installation et utilisation

### Prérequis

- [Python 3.13+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) — gestionnaire de packages moderne


### Installation de uv

**Linux / macOS :**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows :**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Via pip (alternative) :**
```bash
pip install uv
```

> Sur SSPCloud/Onyxia, uv est déjà installé — pas besoin de cette étape.

Après installation, redémarrer son terminal puis vérifier :
```bash
uv --version
```
```bash
# Installer uv si pas encore installé
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Cloner le dépôt

```bash
git clone https://github.com/aristidinahfaifa-afk/G4B_pollenguard_rennes.git
cd G4B_pollenguard_rennes
```

### 2. Installer l'environnement

```bash
uv sync
```

Cette commande crée automatiquement le `.venv`, installe toutes les dépendances et garantit que tout le monde utilise les mêmes versions exactes (via `uv.lock`).

### 3. Lancer le notebook

```bash
uv run jupyter lab
```

Ouvrir `Rapport_final.ipynb` pour reproduire l'analyse complète.

### 4. Lancer l'application Streamlit en local

```bash
uv run streamlit run app/main.py
```

L'application est accessible sur `http://localhost:8501`

---

##  Reproductibilité

Le projet utilise un système de **triple fallback** pour garantir la reproductibilité :

```
1. Cache local  → si data/raw/meteo.csv existe déjà
2. API          → téléchargement depuis Open-Meteo (gratuit, sans clé)
3. S3           → si l'API est indisponible (données de secours sur MinIO SSPCloud)
```

### Ajouter une dépendance

```bash
uv add nom_du_package
git add pyproject.toml uv.lock
git commit -m "add: nom_du_package"
git push
```

### Mettre à jour après un `git pull`

```bash
git pull
uv sync
```

---

## Équipe

Ce projet a été réalisé par :

- **AIFA ARISTIDINA**
- **KENNE YONTA Lesline**
- **ROSE Valentin**

Encadrants : **Julien Pramil** (TP) · **Lino Galiana** (Cours)

---

##  Limites

- Modèles entraînés sur **Rennes uniquement**, non généralisables directement à d'autres villes
- Les données pollen de l'API Open-Meteo sont issues d'un **modèle atmosphérique**, pas de mesures terrain (contrairement au Réseau Sentinelles / RNSA)
- L'API archive publie ses données avec **1 à 2 jours de délai**

---

## Licence

Projet académique — usage éducatif uniquement.  
Données : [Open-Meteo](https://open-meteo.com/) (CC BY 4.0)

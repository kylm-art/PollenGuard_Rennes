# G4B_acces_equip_medical_france
Analyse des inégalités territoriales d’accès aux soins en France à l’échelle communale, à partir des équipements de santé (BPE), des données socio-économiques (FILOSOFI) et des temps d’accès aux hôpitaux.


# Installation et utilisation du projet

## 1. Cloner le dépôt

```bash
git clone https://github.com/aristidinahfaifa-afk/G4B_acces_equip_medical_france.git
cd G4B_acces_equip_medical_france
```



## 2. Installer l’environnement

Ce projet utilise uv pour gérer les dépendances.

```bash
uv sync
```

Cette commande :

* crée l’environnement virtuel (.venv)
* installe toutes les dépendances nécessaires
* garantit que tout le monde utilise les mêmes versions


## 3. Activer l’environnement

```bash
source .venv/bin/activate
```


## 4. Ajouter une nouvelle dépendance (développeurs)

```bash
uv add nom_du_package
```

Puis :

```bash
git add pyproject.toml uv.lock
git commit -m "add dependency"
git push
```


## 5. Mettre à jour le projet

Après un pull :

```bash
git pull
uv sync
```

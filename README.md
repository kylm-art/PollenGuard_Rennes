## G4B_Projet Allergo-Predict : Modélisation et Prédiction du Risque Allergique à Rennes

---
### Description du projet
Ce projet  a pour objectif d'anticiper les pics de concentration pollinique dans l'atmosphère de la ville de Rennes en s'appuyant sur les corrélations entre les conditions météorologiques et la biologie végétale. En exploitant des données historiques de haute précision (horaires), nous développons un modèle de Machine Learning capable de classifier le niveau de risque allergique pour la population.

### Méthodologie et Sources de données
Le projet repose sur la fusion de deux sources de données distinctes via l'API Open-Meteo :
* **Données Biologiques** : Concentrations de pollen de bouleau et de graminées (exprimées en grains/$m^3$).
* **Données Météorologiques** : Température à 2 mètres, vitesse du vent à 10 mètres et précipitations cumulées.

### Construction de la Variable Cible (Target)
La variable cible suitla logique de santé publique suivante:


1.  **Agrégation Temporelle** : Les données horaires sont agrégées à l'échelle journalière en retenant la valeur maximale de concentration rencontrée sur 24 heures (le pic d'exposition étant le facteur déclencheur des symptômes).
2.  **Calcul du Risque Global** : Nous calculons l'exposition maximale entre les deux taxons (Bouleau et Graminées) afin d'obtenir un indicateur de risque unique.
3.  **Classification Clinique** : La variable est ensuite discrétisée en trois classes (0, 1 et 2). 

La classification est ensuite faite en se basant sur les seuils recommandés par le **RNSA (Réseau National de Surveillance Aérobiologique)** :
    * **Niveau 0 (Faible)** : Concentration $\leq$ 30 grains/$m^3$.
    * **Niveau 1 (Modéré)** : Concentration comprise entre 31 et 80 grains/$m^3$.
    * **Niveau 2 (Élevé)** : Concentration $>$ 80 grains/$m^3$. 


### Variables Explicatives (Features)
Pour maximiser la performance du modèle, nous utilisons :
* Des indicateurs météorologiques directs (température, vent, pluie).
* Des variables de saisonnalité (mois, semaine de l'année).
* Des variables d'inertie biologique (température de la veille et moyennes glissantes sur 3 jours).

---


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

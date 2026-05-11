# Projet Fairness — Chest X-Ray NIH 14

Ce projet étudie les biais potentiels d’un modèle de classification d’images médicales appliqué au dataset **NIH Chest X-Ray 14**. L’objectif est d’analyser si les performances du modèle varient selon certaines caractéristiques des patients, puis de tester plusieurs stratégies pour réduire ces écarts.

Le modèle utilisé est un **ResNet18** entraîné pour réaliser une classification binaire à partir de radiographies thoraciques. Même si le modèle ne prend que l’image en entrée, les métadonnées associées aux images permettent d’étudier l’équité des prédictions selon différents groupes.

## Auteurs

- Malak ELMASRY
- Rima OUCHERIF
- Malak ZIAD BADRAN

## Objectifs du projet

Le projet poursuit trois objectifs principaux :

1. Analyser les déséquilibres présents dans les données médicales.
2. Évaluer les performances du modèle selon différents groupes de patients.
3. Tester des méthodes de correction des biais par pré-traitement et post-traitement.

Les attributs sensibles étudiés sont :

- le genre du patient ;
- le groupe d’âge ;
- la position de vue de la radiographie, notamment `AP` et `PA`.

## Dataset

Le projet utilise le dataset **NIH Chest X-Ray 14**, composé de radiographies thoraciques accompagnées de métadonnées telles que :

- l’âge du patient ;
- le genre du patient ;
- la position de vue de l’image ;
- le label de classification ;
- le split `train` / `valid`.

Dans le notebook, l’âge est regroupé en trois classes :

- `<40` ;
- `40-60` ;
- `>60`.

Cette étape permet de comparer plus facilement les performances du modèle entre différentes catégories de patients.

## Méthodologie

### 1. Analyse exploratoire des données

Une première analyse descriptive est réalisée afin d’identifier les déséquilibres du dataset. Elle permet notamment d’observer la distribution des labels selon l’âge, le genre et la position de vue.

L’analyse montre que l’âge est l’attribut le plus fortement lié au label : les patients de plus de 60 ans présentent un taux de maladie plus élevé que les patients plus jeunes.

### 2. Entraînement du modèle

Le modèle utilisé est un **ResNet18**. Plusieurs versions du modèle sont entraînées avec différentes stratégies de pondération des exemples.

Les cinq stratégies comparées sont :

| Stratégie | Description |
|---|---|
| `W1_BASELINE` | Tous les exemples ont le même poids. |
| `W2_CLASS` | Rééquilibrage entre les classes `sain` et `malade`. |
| `W3_AGE` | Pondération selon le couple `(label, groupe d’âge)`. |
| `W4_GENDER` | Pondération selon le couple `(label, genre)`. |
| `W5_VIEW` | Pondération selon le couple `(label, position de vue)`. |

Le fichier `train_classifieur.py` utilise un `WeightedRandomSampler`, ce qui permet de modifier la fréquence d’apparition de certains exemples pendant l’entraînement.

### 3. Évaluation des performances et de la fairness

Les modèles sont évalués à l’aide de plusieurs métriques :

- accuracy ;
- balanced accuracy ;
- balanced accuracy par groupe ;
- disparité entre groupes ;
- Equal Opportunity Difference, pour l’analyse du taux de vrais positifs.

La disparité est calculée comme l’écart entre le meilleur et le moins bon groupe pour un attribut donné.

### 4. Pré-traitement

Le pré-traitement repose sur la pondération des exemples avant l’entraînement. L’objectif est de réduire l’impact des déséquilibres présents dans le dataset.

La stratégie `W3_AGE` permet de réduire les disparités liées à l’âge, mais avec une légère baisse de performance globale. À l’inverse, certaines stratégies comme `W2_CLASS` ou `W4_GENDER` obtiennent de meilleures performances globales, mais ne réduisent pas toujours les biais de manière optimale.

### 5. Post-traitement

Le post-traitement consiste à modifier la règle de décision après l’entraînement, sans réentraîner le modèle.

Deux approches sont étudiées :

- l’ajustement global du seuil de décision ;
- l’égalisation des opportunités par groupe d’âge.

L’égalisation des opportunités permet de rapprocher les taux de vrais positifs entre les groupes d’âge, mais entraîne une baisse de balanced accuracy globale.

## Résultats principaux

Les résultats montrent qu’il existe un compromis entre performance et équité.

- L’âge est l’attribut pour lequel les disparités sont les plus marquées.
- La stratégie `W3_AGE` réduit mieux les disparités liées à l’âge.
- La stratégie `W4_GENDER` offre un bon compromis entre performance globale et équité.
- La stratégie `W2_CLASS` obtient de bonnes performances globales, mais peut conserver une disparité importante selon l’âge.
- Le post-traitement améliore certains critères d’équité, mais au prix d’une baisse de performance.

Exemples de résultats observés dans le notebook :

| Modèle | Balanced Accuracy | Disparité Genre | Disparité Âge | Disparité ViewPos |
|---|---:|---:|---:|---:|
| `W1_BASELINE` | 0.6706 | 0.0349 | 0.0320 | 0.0067 |
| `W2_CLASS` | 0.6706 | 0.0064 | 0.0796 | 0.0080 |
| `W3_AGE` | 0.6753 | 0.0356 | 0.0446 | 0.0105 |
| `W4_GENDER` | 0.6881 | 0.0154 | 0.0629 | 0.0079 |
| `W5_VIEW` | 0.6841 | 0.0068 | 0.0492 | 0.0151 |

Ces résultats confirment qu’un modèle plus performant globalement n’est pas nécessairement le plus équitable entre sous-groupes.

## Structure du projet

Une structure possible du dépôt GitHub est la suivante :

```text
.
├── projet_notebook_FINAL_FINAL.ipynb
├── train_classifieur.py
├── DATA/
│   ├── metadata.csv
│   └── images/
├── expe_log/
│   ├── W1_BASELINE/
│   ├── W2_CLASS/
│   ├── W3_AGE/
│   ├── W4_GENDER/
│   └── W5_VIEW/
└── README.md
```

Le dossier `DATA` contient les données utilisées par le projet. Le dossier `expe_log` contient les résultats des entraînements, les modèles sauvegardés et les fichiers de prédictions.

## Installation

Le projet a été réalisé sur **Google Colab**.

Les principales bibliothèques utilisées sont :

```bash
pip install torchmetrics pytorch-lightning seaborn fairlearn aif360
```

Les bibliothèques Python utilisées dans le notebook incluent notamment :

- `pandas` ;
- `numpy` ;
- `matplotlib` ;
- `seaborn` ;
- `scikit-learn` ;
- `fairlearn` ;
- `aif360` ;
- `pytorch-lightning` ;
- `torchmetrics`.

## Utilisation

Pour exécuter le projet :

1. Ouvrir le notebook dans Google Colab.
2. Monter Google Drive.
3. Vérifier que les chemins vers le dossier du projet, les données et les logs sont corrects.
4. Installer les bibliothèques nécessaires.
5. Lancer les cellules dans l’ordre.

Dans le notebook, les chemins principaux sont définis ainsi :

```python
PROJECT_DIR = "/content/drive/MyDrive/proj_fairness"
DATA_DIR = os.path.join(PROJECT_DIR, "DATA")
CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")
LOG_BASE = os.path.join(PROJECT_DIR, "expe_log")
```

## Limites

Ce projet présente plusieurs limites :

- le modèle utilise uniquement les images et non les métadonnées ;
- les attributs sensibles peuvent être appris implicitement à partir des radiographies ;
- les méthodes de correction utilisées restent simples ;
- l’amélioration de l’équité peut réduire la performance globale ;
- l’égalisation des opportunités nécessite de connaître le groupe d’âge au moment de l’inférence.

Des méthodes plus avancées, comme l’in-processing ou l’ajout de contraintes explicites de fairness pendant l’apprentissage, pourraient être explorées dans un travail futur.

## Conclusion

Ce projet montre que l’évaluation d’un modèle médical ne doit pas se limiter à sa performance globale. Même lorsqu’un modèle obtient une bonne accuracy, il peut présenter des écarts importants entre différents groupes de patients.

L’analyse réalisée met en évidence un biais principalement lié à l’âge. Les stratégies de pondération et de post-traitement permettent de réduire certaines disparités, mais elles introduisent aussi un compromis entre performance et équité. Dans un contexte médical, ce compromis doit être étudié avec attention, car la réduction des inégalités entre patients peut être aussi importante que l’amélioration de la performance moyenne.

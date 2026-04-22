
# 🏠 Housing Regression MLE - End-to-End Pipeline

> **Prédiction de prix immobiliers aux États-Unis : De la donnée brute au déploiement AWS.**

Ce projet démontre la construction d'un système de Machine Learning complet, robuste et prêt pour la production. Il couvre tout le cycle de vie MLOps : ingestion des données, ingénierie des features, entraînement de modèles (XGBoost), suivi d'expériences (MLflow), conteneurisation (Docker) et déploiement cloud automatisé (AWS ECS + GitHub Actions).

---

## 🎯 Objectifs du Projet

1.  **Fiabilité des Données** : Prévenir le *data leakage* via des splits temporels stricts et valider la qualité avec Great Expectations.
2.  **Reproductibilité** : Garantir que le modèle s'exécute identiquement partout grâce à Docker et `uv`.
3.  **Automatisation (MLOps)** : Mettre en place un pipeline CI/CD qui teste, build et déploie le modèle à chaque push sur GitHub.
4.  **Accessibilité** : Rendre le modèle utilisable via une API REST (FastAPI) et une interface utilisateur interactive (Streamlit).

---

## 🏗️ Architecture du Projet

Le projet suit une architecture modulaire séparant clairement l'expérimentation de la production.

```
Regression_ML_EndtoEnd/
│
├── 📁 src/                      # Code de production (Modulaire)
│   ├── 📁 feature_pipeline/     # Chargement, nettoyage, encoding
│   ├── 📁 training_pipeline/    # Entraînement, tuning (Optuna), tracking (MLflow)
│   ├── 📁 inference_pipeline/   # Prédiction batch & real-time
│   └── 📁 api/                  # Endpoint FastAPI
│
├── 📁 notebooks/                # 🧪 Expérimentations (EDA, Baseline, Tuning)
├── 📁 tests/                    # ✅ Tests unitaires (Pytest) & Smoke tests
├── 📁 configs/                  # ⚙️ Configurations YAML (App, MLflow, GE)
├── 📁 .github/workflows/        # 🤖 CI/CD Pipeline (GitHub Actions)
│
├── 📁 data/                     # Données (Ignoré par Git, stocké sur S3 in prod)
│   ├── raw/                     # Données brutes & splits temporels
│   └── processed/               # Données nettoyées & features engineering
│
├── 📁 models/                   # 🧠 Artefacts sauvegardés (.pkl)
├── 📄 app.py                    # 🎨 Interface Streamlit (Frontend)
├── 📄 Dockerfile                # 🐳 Image pour l'API Backend
├── 📄 Dockerfile.streamlit      # 🐳 Image pour l'UI Frontend
├── 📄 pyproject.toml            # 📦 Gestion des dépendances (uv)
└── 📄 README.md                 # 📖 Documentation
```

### 🔑 Composants Clés

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Data Processing** | Pandas, Great Expectations | Nettoyage, validation de schéma, détection d'outliers. |
| **Feature Engineering** | Custom Encoders | Frequency Encoding (ZipCode), Target Encoding (City), Lat/Long mapping. |
| **Modeling** | XGBoost, Optuna, Scikit-Learn | Régression, optimisation bayésienne des hyperparamètres. |
| **Tracking** | MLflow | Versioning des modèles, métriques (MAE, RMSE, R²) et paramètres. |
| **API** | FastAPI, Uvicorn | Service REST pour les prédictions en temps réel. |
| **UI** | Streamlit, Plotly | Dashboard interactif pour visualiser les prédictions vs réels. |
| **Infrastructure** | Docker, AWS (S3, ECR, ECS, ALB) | Conteneurisation, stockage, orchestration serverless. |
| **CI/CD** | GitHub Actions | Automatisation des tests, build Docker et déploiement AWS. |

---
### roadmap

```
PHASE 1 : EXPÉRIMENTATION (Notebooks)
├─ 00_data_split.ipynb → Split temporel (train/eval/holdout)
├─ 01_EDA_cleaning.ipynb → Nettoyage + mapping ville→lat/long
├─ 02_feature_eng_encoding.ipynb → Encoding + features temporelles
├─ 03_baseline.ipynb → Premier modèle simple (DummyRegressor)
├─ 04-05_modèles.ipynb → XGBoost + comparaison
└─ 06_tuning.ipynb → Optuna + MLflow

PHASE 2 : PRODUCTION (src/)
├─ feature_pipeline/ → load.py, preprocess.py, encode.py
├─ training_pipeline/ → train.py, tune.py, evaluate.py
├─ inference_pipeline/ → predict.py
└─ api/ → FastAPI endpoints

PHASE 3 : INDUSTRIALISATION
├─ tests/ → pytest pour valider chaque pipeline
├─ Dockerfile → Empaqueter l'API
├─ Dockerfile.streamlit → Empaqueter l'UI
├─ .github/workflows/ci.yml → Automatiser le déploiement
└─ AWS → Héberger en production

PHASE 4 : UTILISATION
├─ FastAPI → /predict endpoint pour les devs
└─ Streamlit → Dashboard interactif pour les utilisateurs finaux
```

---

## 🚀 Installation & Setup Local

### Prérequis
*   Python 3.11+
*   [uv](https://docs.astral.sh/uv/) (Gestionnaire de packages ultra-rapide)
*   Docker Desktop
*   Compte AWS (pour le déploiement)

### 1. Cloner le repository
```bash
git clone https://github.com/hamzaDn/housing-mle.git
cd housing-mle
```

### 2. Installer les dépendances
Nous utilisons `uv` pour une gestion rapide et reproductible des environnements.
```bash
# Initialiser l'environnement virtuel et installer les deps
uv sync

# Activer l'environnement
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Configuration des Données
Téléchargez les datasets suivants et placez-les dans `data/raw/` :
1.  **HouseTS Dataset** (Kaggle) : Renommez-le `house_ts_original.csv`.
2.  **US Metros Dataset** (SimpleMaps) : Renommez-le `us_metros.csv`.

### 4. Exécution Locale

#### A. Pipeline de Features & Training
Exécutez les notebooks dans l'ordre ou utilisez les scripts Python dans `src/` :
```bash
# Exemple pour lancer le training pipeline
python src/training_pipeline/train.py
```

#### B. Lancer l'API (Backend)
```bash
uvicorn src.api.main:app --reload
# accessible sur http://127.0.0.1:8000/docs
```

#### C. Lancer l'Interface UI (Frontend)
```bash
streamlit run app.py
# accessible sur http://127.0.0.1:8501
```

---

## 🧪 Testing & Qualité

Le projet inclut une suite de tests automatisés pour garantir la robustesse du code.

```bash
# Lancer tous les tests
pytest tests/ -v

# Lancer les tests avec couverture
pytest tests/ --cov=src
```

*   **Unit Tests** : Vérifient chaque fonction du pipeline (load, clean, encode).
*   **Smoke Tests** : Vérifient que le pipeline end-to-end produit bien des prédictions numériques.
*   **Data Quality** : Great Expectations valide l'intégrité des données avant traitement.

---

## ☁️ Déploiement AWS (CI/CD)

Le déploiement est entièrement automatisé via **GitHub Actions**. À chaque push sur la branche `main` :

1.  **Test** : Exécution de `pytest`.
2.  **Build** : Construction des images Docker (API & Streamlit).
3.  **Push** : Envoi des images vers **Amazon ECR** (Elastic Container Registry).
4.  **Deploy** : Mise à jour des services **Amazon ECS Fargate**.
5.  **Access** : L'application est accessible via l'**Application Load Balancer (ALB)**.

### Secrets GitHub Requis
Pour que le CI/CD fonctionne, configurez ces secrets dans votre repo GitHub (`Settings > Secrets and variables > Actions`) :
*   `AWS_ACCESS_KEY_ID`
*   `AWS_SECRET_ACCESS_KEY`
*   `AWS_REGION` (ex: `eu-west-2`)
*   `ECR_REGISTRY`
*   `ECR_REPOSITORY_API`
*   `ECR_REPOSITORY_STREAMLIT`

---

## 📊 Résultats & Métriques

Le modèle final (XGBoost optimisé avec Optuna) atteint les performances suivantes sur le set de Holdout (2022-2023) :

*   **R² Score** : ~0.96 (Explique 96% de la variance des prix)
*   **MAE (Mean Absolute Error)** : ~$31,000
*   **RMSE (Root Mean Squared Error)** : ~$70,000

*Ces métriques sont suivies et comparées historiquement via MLflow.*

---

## 💡 Leçons Apprises & Best Practices

1.  **Split Temporel** : Crucial pour les séries temporelles. Un split aléatoire aurait introduit un *data leakage* majeur (le modèle aurait "vu" le futur).
2.  **Encodage Géographique** : Remplacer les noms de villes par leur Latitude/Longitude a permis au modèle de capturer la proximité géographique sans exploser la dimensionnalité (comme l'aurait fait un One-Hot Encoding).
3.  **Modularité** : Séparer le code en pipelines (`feature`, `training`, `inference`) facilite la maintenance et le débogage par rapport aux notebooks monolithiques.
4.  **Infra as Code** : Utiliser GitHub Actions pour le déploiement élimine les erreurs humaines ("ça marche sur ma machine") et assure une reproductibilité totale.

---

## 👤 Auteur

**[Hamza MOUFID]**
*Data Scientist / ML Engineer*

*   🔗 [LinkedIn](T//)
*   💻 [GitHub](//)



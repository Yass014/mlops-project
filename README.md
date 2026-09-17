# Projet MLOps - California Housing

Pipeline MLOps de bout en bout réalisé pour le cours de M2 Campus Cyber.

## Structure
- `src/` : Pipeline d'entraînement et tracking MLflow.
- `api/` : API FastAPI pour servir le modèle.
- `configs/` : Configuration centralisée (YAML).
- `Makefile` : Orchestration des tâches.

## Guide d'utilisation

Pour cloner et lancer le projet localement :

### 1. Cloner le projet
\`\`\`bash
git clone https://github.com/Yass014/mlops-project.git
cd mlops-project
\`\`\`

### 2. Installer les dépendances
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 3. Entraîner le modèle
\`\`\`bash
python src/train.py
\`\`\`
*(Ou via le Makefile : `make train`)*

### 4. Lancer l'API FastAPI
\`\`\`bash
uvicorn api.app:app --reload
\`\`\`
*(Ou via le Makefile : `make run-api`)*

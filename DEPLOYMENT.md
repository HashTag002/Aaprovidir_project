# Deployment notes - AaPROVIDIR

## Objectif

Preparer le projet pour un hebergement web propre avec Dash comme interface principale.

## Service applicatif

- Point d'entree actuel en developpement : `env/bin/python3 run.py`.
- Dash sert les dashboards sur le port `8050`.
- Django sert la page d'accueil secondaire sur le port `8000`.
- Pour une premiere mise en ligne simple, exposer Dash comme service principal.

## Dependances systeme et Python

Installer les dependances Python depuis `requirements.txt` :

```bash
python3 -m venv env
env/bin/python3 -m pip install -r requirements.txt
```

Les dependances lourdes a prevoir dans l'image ou l'environnement :

- `xgboost`
- `tensorflow`
- `scikit-learn`
- `statsmodels`

## Variables et stockage

- Les donnees sont dans `data/Dataset.csv`.
- Les modeles entraines sont dans `models/`.
- Les fichiers `.joblib` doivent etre conserves avec l'application.
- Le LSTM est sauvegarde en `.keras` et son preprocesseur en `.joblib`.

## Commandes utiles

Controle avant de deployer :

```bash
env/bin/python3 manage.py check
env/bin/python3 manage.py test
env/bin/python3 modeling/model_study.py --quick --include-lstm --window-size 12
```

## Recommandation d'hebergement

Pour un prototype :

1. Utiliser une plateforme qui accepte les apps Python longues a demarrer et les artefacts volumineux (`models/`).
2. Preinstaller les dependances lourdes dans l'image pour eviter les timeouts au demarrage.
3. Exposer Dash en priorite.
4. Garder Django optionnel pour l'accueil et l'administration.

Pour une version production :

- separer l'entrainement des modeles du serveur web ;
- stocker les modeles dans un stockage persistant ;
- ajouter un job planifie pour reentrainer les modeles ;
- ajouter une API externe de recherche marche si l'assistant investisseur doit utiliser Internet en temps reel.

# Rapport des modifications - AaPROVIDIR Analytics

## Objectif de la mise a jour

Ajouter une experience applicative plus complete autour du projet PESTEL :

- une page d'accueil principale claire ;
- un dashboard descriptif EDA enrichi par une conclusion automatique ;
- un dashboard de previsions et recommandations compatible avec les modeles `.joblib` ;
- une interface plus propre, epuree et professionnelle ;
- un rapport versionne a actualiser a chaque evolution du projet.

## Modifications realisees

### 1. Page d'accueil principale

- Ajout d'une page d'accueil Dash accessible depuis `http://localhost:8050/`.
- Modernisation de la page d'accueil Django accessible depuis `http://localhost:8000/` et `http://localhost:8000/dashboard/`.
- Ajout de deux redirections principales :
  - `http://localhost:8050/descriptif` pour le dashboard descriptif ;
  - `http://localhost:8050/previsions` pour le dashboard de previsions et recommandations.

### 2. Dashboard descriptif

- Transformation du dashboard Dash en application multi-pages.
- Conservation des modules EDA existants :
  - synthese ;
  - audit qualite ;
  - analyse prix ;
  - facteurs PESTEL ;
  - segmentation ;
  - chocs et anomalies.
- Ajout d'un onglet final `CONCLUSION`.
- Generation automatique d'une conclusion narrative basee sur :
  - la periode filtree ;
  - le produit et la region selectionnes ;
  - la variable la plus correlee au prix ;
  - la saisonnalite STL de periode 12 mois ;
  - le nombre d'anomalies detectees ;
  - le nombre d'anomalies liees aux chocs encodes ;
  - le cluster du produit ;
  - les features prioritaires recommandees.
- Amelioration de la conclusion en synthese decisionnelle :
  - diagnostic de tendance ;
  - niveau de risque analytique ;
  - signaux cles ;
  - actions recommandees ;
  - cartes de lecture rapide.
- Amelioration de l'audit qualite avec score global, valeurs manquantes, doublons, couverture temporelle et graphe de completude toujours visible.
- Amelioration de la segmentation :
  - choix utilisateur du nombre de clusters KMeans ;
  - PCA conservee pour la visualisation ;
  - tableau de profil des clusters.

### 3. Dashboard previsions et recommandations

- Ajout d'une page Dash dediee aux previsions.
- Ajout d'une detection automatique des fichiers `.joblib` dans le dossier `models`.
- Ajout d'une interface de selection :
  - produit ;
  - region ;
  - modele `.joblib` ;
  - horizon de prevision.
- Ajout d'un rendu de prevision avec :
  - historique recent ;
  - trajectoire prevue ;
  - prix observe ;
  - prix prevu a horizon ;
  - nombre de modeles detectes.
- Ajout de recommandations automatiques selon :
  - la variation attendue ;
  - les chocs recents ;
  - la variable PESTEL la plus associee au prix.
- Si aucun modele `.joblib` n'est present, la page affiche une projection indicative et signale clairement l'absence de modele.
- Correction de l'alignement des features pour les modeles sklearn sans noms de colonnes :
  - exclusion des colonnes d'aide calculees par l'application comme `Score_Choc` ;
  - utilisation du nombre de features attendu par le modele ;
  - prevention du repli indicatif lorsque `regression_model.joblib` attend les 21 variables d'origine.
- Ajout d'une page `Tests modeles` accessible via `http://localhost:8050/modeles` :
  - evaluation des modeles `.joblib` charges ;
  - comparaison avec une baseline `Prix_T-1` ;
  - calcul MAE, RMSE, MAPE et R2 ;
  - graphe prix reels vs previsions.

### 4. Style et ergonomie

- Refonte de `assets/style.css`.
- Ajout d'une navigation superieure commune.
- Amelioration des cartes KPI, filtres, onglets, sections, listes et graphes.
- Harmonisation de la page Django avec le style Dash.
- Rendu plus sobre, lisible et professionnel.

### 5. Structure projet

- Creation du dossier `models/`.
- Ajout de `models/.gitkeep` pour conserver le dossier dans Git.
- Mise a jour de `run.py` pour afficher les nouvelles URLs utiles au demarrage.
- Ajout de l'URL racine Django `/`.

## Fichiers ajoutes ou modifies

- `dashboard/dash_app.py`
- `assets/style.css`
- `dashboard/templates/dashboard/index.html`
- `aaprovidir_project/urls.py`
- `run.py`
- `models/.gitkeep`
- `RAPPORT_MODIFICATIONS.md`
- `dashboard/tests.py`

## Points a maintenir lors des prochaines modifications

- Actualiser ce rapport apres chaque ajout fonctionnel ou changement important.
- Placer les modeles de prevision sous `models/` avec l'extension `.joblib`.
- Documenter le nom du modele, ses features d'entree et sa variable cible si plusieurs modeles sont ajoutes.
- Si un modele impose une structure de features stricte, adapter la preparation des donnees dans `dashboard/dash_app.py`.

## Etat courant

- Accueil principal : ajoute.
- Dashboard descriptif : enrichi avec conclusion automatique.
- Dashboard previsions : ajoute avec detection `.joblib`.
- Style global : refondu.
- Verification technique : compilation Python, checks Django, tests Django et validation directe des composants Dash executes.
- Verification modele `.joblib` : test temporaire realise avec un modele factice, puis modele supprime.
- Verification manuelle : accueil, dashboard descriptif, onglet conclusion, onglets EDA et page previsions testes dans le navigateur.
- Correction appliquee pendant la verification : remplacement des onglets Bootstrap par des onglets Dash natifs afin de garantir le declenchement des callbacks.
- Correction appliquee au lanceur : ajout de `--noreload` au serveur Django pour eviter l'arret premature de `run.py`.
- Nouvelle iteration : conclusion decisionnelle, audit qualite renforce, segmentation KMeans parametrable, correction du mismatch 21/22 features et page de test/comparaison des modeles.

## Iteration modelisation avancee

### Dossier d'etude des modeles

- Ajout du dossier `modeling/`.
- Ajout du fichier `modeling/model_study.py` pour entrainer et comparer :
  - regression lineaire multiple ;
  - Random Forest ;
  - XGBoost ;
  - LSTM TensorFlow.
- Ajout d'une validation temporelle :
  - split holdout temporel ;
  - baseline `Prix_T-1` ;
  - `TimeSeriesSplit` pour la cross validation ;
  - `GridSearchCV` pour optimiser Random Forest et XGBoost.
- Sauvegarde des resultats dans `models/` :
  - `regression_model.joblib` ;
  - `random_forest_model.joblib` ;
  - `xgboost_model.joblib` ;
  - `best_model.joblib` ;
  - `lstm_model.keras` ;
  - `lstm_preprocessor.joblib` ;
  - `model_study_metrics.csv` ;
  - `model_study_results.json` ;
  - `best_model_metadata.json`.

### Resultats de l'entrainement rapide

- Baseline `Prix_T-1` : MAE holdout 199.44.
- Regression lineaire multiple : MAE holdout 145.76, meilleur modele `.joblib` et modele retenu comme `best_model.joblib`.
- Random Forest : MAE holdout 282.42, conserve pour comparaison mais moins performant que la baseline.
- XGBoost : MAE holdout 292.90, conserve pour comparaison mais moins performant que la baseline.
- LSTM : MAE holdout 1189.45 en entrainement rapide CPU, conserve comme prototype sequence mais non retenu comme meilleur modele.

### Problemes rencontres et resolutions

- Probleme : `xgboost` et `tensorflow` etaient absents de l'environnement.
  - Resolution : installation des dependances et ajout dans `requirements.txt`.
- Probleme : TensorFlow signale l'absence de GPU/CUDA.
  - Resolution : entrainement LSTM effectue sur CPU ; les logs CUDA sont non bloquants.
- Probleme : les modeles sklearn sans noms de colonnes peuvent recevoir 22 features a cause de `Score_Choc`, colonne calculee par l'application.
  - Resolution : exclusion des colonnes d'aide dans la preparation des features et validation par tests.
- Probleme : `lstm_preprocessor.joblib` apparaissait comme modele dans le dashboard.
  - Resolution : filtrage des fichiers `*_preprocessor.joblib` dans la detection des modeles.
- Probleme : Random Forest, XGBoost et LSTM ne battent pas la baseline sur l'entrainement rapide.
  - Resolution : conservation des metriques pour analyse, sauvegarde du meilleur modele `.joblib`, et script parametrable pour relancer une recherche plus longue sans `--quick`.

## Iteration correction style

- Probleme : les pages Dash restaient visuellement basiques car le fichier `assets/style.css` n'etait pas servi par Dash (`/assets/style.css` retournait 404).
  - Cause : le dossier d'assets etait a la racine du projet alors que l'application Dash est declaree dans `dashboard/dash_app.py`.
  - Resolution : declaration explicite de `assets_folder` et `assets_url_path` dans l'initialisation Dash.
- Amelioration complementaire :
  - renforcement des styles globaux ;
  - cartes plus modernes ;
  - hero gradient plus visible ;
  - dropdowns, tableaux, graphes et sliders mieux stylises ;
  - verification navigateur sur accueil, dashboard descriptif et dashboard previsions.

## Iteration comparaison et selection des modeles

- Ajout d'un selecteur multi-modeles sur la page `Tests modeles`.
- Remplacement du multi-select par une checklist visible afin de rendre la selection plus simple et plus fiable.
- Le meilleur modele n'est plus uniquement le plus precis : il est choisi avec un score d'equilibre combinant :
  - 60 % precision, calculee avec le ratio entre la meilleure MAE et la MAE du modele ;
  - 40 % explicabilite, favorisant les modeles lineaires puis les ensembles d'arbres, puis les reseaux de neurones.
- Justification affichee dans la page `Tests modeles` pour expliquer pourquoi le modele est retenu.
- La page `Previsions` utilise par defaut le modele recommande par ce compromis precision-explicabilite.
- Probleme UX : les menus deroulants de la page `Previsions` pouvaient etre masques par les KPI.
  - Resolution : priorite visuelle plus forte pour la carte des filtres, menus dropdown au-dessus du contenu et espacement vertical renforce.
- Probleme UX : le slider de taille de test pouvait sembler changer pendant le chargement de la page.
  - Resolution : remplacement par un menu deroulant stable `10 % / 20 % / 30 % / 40 %`.
- Probleme UX : la page `Previsions` pouvait afficher un etat initial incomplet avant que les callbacks remplissent la region.
  - Resolution : initialisation directe des options et de la valeur region dans le layout.

## Iteration fenetres temporelles et assistant investisseur

- Correction de la modelisation :
  - MLP, Random Forest et XGBoost sont maintenant entraines sur des fenetres temporelles ;
  - la fenetre par defaut est de 12 mois pour capturer la saisonnalite annuelle ;
  - chaque ligne d'apprentissage utilise les 12 mois passes pour predire le mois suivant ;
  - le LSTM utilise la meme taille de fenetre au niveau de la standardisation et de la sequence d'entree.
- Ajout d'un controle `Fenetre historique` dans la page `Tests modeles`.
- Ajout de `mlp_model.joblib`.
- Ajout de `feature_importance.csv` dans `models/`.
- Reduction de dimension :
  - selection des 10 features les plus importantes avant creation des fenetres ;
  - PCA appliquee aux modeles lineaires/MLP fenetres lorsque pertinent ;
  - la regression multiple de production reste sans PCA pour conserver l'explicabilite des coefficients.
- Resultat courant :
  - `regression_model.joblib` reste le meilleur modele de production car il bat la baseline et reste explicable ;
  - MLP, Random Forest, XGBoost et LSTM sont conserves pour comparaison et optimisation future.
- Dashboard :
  - la segmentation affiche maintenant les features par ordre d'importance ;
  - la page `Previsions` contient un assistant investisseur ;
  - l'assistant peut retourner les 5 features cles ou proposer des filieres a analyser pour 2027 a partir des prix prevus et de la stabilite historique.
- Hebergement :
  - ajout de `DEPLOYMENT.md` ;
  - recommandation de separer l'entrainement des modeles du serveur web en production ;
  - prevoir une API externe de recherche marche si l'assistant investisseur doit utiliser Internet en temps reel.

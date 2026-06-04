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

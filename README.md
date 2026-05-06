# PESTEL Analytics — Marchés Agricoles

Système de prédiction des fluctuations de prix basé sur les facteurs PESTEL.

## Architecture du projet

```
pestel_dashboard/
├── data/
│   └── pestel_agricole.csv      ← Données PESTEL (60 mois, 5 produits, 23 variables)
├── dashboard/                   ← App Django
│   ├── dash_app.py              ← MODULE 1 : Dashboard interactif (Dash)
│   ├── views.py                 ← Vues Django + endpoint upload CSV
│   └── urls.py
├── pestel_project/              ← Config Django
├── templates/dashboard/
│   └── index.html               ← Shell Django avec navigation modules
├── requirements.txt
├── run.py                       ← Lanceur unifié
└── manage.py
```

## Modules planifiés

| Module | Statut | Description |
|--------|--------|-------------|
| 1 — INPUT Dashboard | ✅ Fait | Visualisation données d'entrée CSV |
| 2 — ML Model | 🔲 À faire | Entraînement / évaluation modèle |
| 3 — Prévisions | 🔲 À faire | Dashboard des prédictions de prix |

## Installation

```bash
pip install -r requirements.txt
python manage.py migrate
python run.py
```

→ Dashboard Dash : http://localhost:8050  
→ App Django : http://localhost:8000/dashboard/

## Structure CSV attendue

La **première ligne** est l'en-tête. Colonnes minimales recommandées :

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | YYYY-MM | Période |
| `produit` | str | Nom du produit |
| `prix_usd_tonne` | float | Variable cible |
| + variables PESTEL | float | Indicateurs politiques, économiques, etc. |

## Facteurs PESTEL modélisés

- **Politique** : stabilité, politique commerciale, subventions
- **Économique** : PIB, inflation, taux de change, demande
- **Social** : population, urbanisation, revenu
- **Technologique** : agritech, rendement, irrigation
- **Environnemental** : température, précipitations, superficie, sécheresse
- **Légal** : réglementation export, normes phytosanitaires, accords commerciaux

## Visualisations (Module 1)

1. **KPI Cards** — Prix moyen, inflation, PIB, anomalie temp., nb observations
2. **Courbe de prix** — Évolution temporelle par produit
3. **Donut** — Répartition prix moyens par produit
4. **Séries PESTEL** — Indicateurs dans le temps
5. **Heatmap corrélation** — Matrice complète PESTEL × prix
6. **Histogramme + KDE** — Distribution de n'importe quel indicateur
7. **Scatter + tendance** — Indicateur vs Prix avec droite de régression
8. **Radar** — Profil PESTEL normalisé par produit
9. **Tableau filtrable** — Données brutes avec tri et recherche
# Aaprovidir_project
# Aaprovidir_project

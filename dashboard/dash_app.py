from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, dash_table, html
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


# --- CONFIGURATION & THEME -------------------------------------------------
DS_BLUE = "#0f3d5e"
DS_BLUE_LIGHT = "#176b87"
DS_ACCENT = "#2bb3a3"
DS_BG = "#f6f8fb"
DS_TEXT = "#1f2937"
DS_MUTED = "#64748b"
DS_DANGER = "#d94848"
DS_WARNING = "#f59e0b"
DS_SUCCESS = "#16a34a"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Dataset.csv"
MODELS_DIR = BASE_DIR / "models"
TARGET_COL = "Prix_Vente_FCFA_kg"
FORECAST_HELPER_COLUMNS = {"Score_Choc"}
MODEL_METRICS_PATH = MODELS_DIR / "model_study_metrics.csv"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
PRECISION_WEIGHT = 0.60
EXPLAINABILITY_WEIGHT = 0.40

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder=str(BASE_DIR / "assets"),
    assets_url_path="/assets",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "AaPROVIDIR Analytics"


# --- DATA ------------------------------------------------------------------
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, sep=";")
        df["Date"] = pd.to_datetime(df["Date"])
        choc_cols = [c for c in df.columns if c.startswith("Choc_")]
        df["Score_Choc"] = df[choc_cols].sum(axis=1) if choc_cols else 0
        return df
    except Exception as exc:
        print(f"Erreur chargement : {exc}")
        return pd.DataFrame()


df_full = load_data()


def available_products():
    if df_full.empty or "Produit_ID" not in df_full:
        return []
    return sorted(df_full["Produit_ID"].dropna().unique())


def default_product():
    products = available_products()
    return "Maïs" if "Maïs" in products else (products[0] if products else None)


def date_bounds():
    if df_full.empty:
        today = pd.Timestamp.today()
        return today, today
    return df_full["Date"].min(), df_full["Date"].max()


# --- SHARED UI -------------------------------------------------------------
def nav_link(label, href, active=False):
    class_name = "nav-link-clean active" if active else "nav-link-clean"
    return dcc.Link(label, href=href, className=class_name)


def top_nav(active="home"):
    return html.Div(
        className="top-nav",
        children=dbc.Container(
            fluid=True,
            children=[
                dcc.Link(
                    "AaPROVIDIR Analytics",
                    href="/",
                    className="brand-link",
                ),
                html.Div(
                    className="nav-actions",
                    children=[
                        nav_link("Accueil", "/", active == "home"),
                        nav_link("Dashboard descriptif", "/descriptif", active == "descriptif"),
                        nav_link("Prévisions", "/previsions", active == "previsions"),
                        nav_link("Tests modèles", "/modeles", active == "modeles"),
                    ],
                ),
            ],
        ),
    )


def page_header(kicker, title, subtitle):
    return html.Div(
        className="page-header",
        children=[
            html.Div(kicker, className="eyebrow"),
            html.H1(title),
            html.P(subtitle, className="lead-text"),
        ],
    )


def section_card(title, children, subtitle=None, class_name=""):
    return dbc.Card(
        className=f"section-card {class_name}".strip(),
        children=[
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H4(title, className="section-title"),
                            html.P(subtitle, className="section-subtitle") if subtitle else None,
                        ],
                        className="section-heading",
                    ),
                    children,
                ]
            )
        ],
    )


def kpi_card(label, value, color=DS_BLUE, helper=None):
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(label, className="kpi-label"),
                    html.H3(f"{value}", className="kpi-value", style={"color": color}),
                    html.Div(helper, className="kpi-helper") if helper else None,
                ]
            )
        ],
        className="kpi-card h-100",
    )


def graph_card(title, figure, subtitle=None):
    return section_card(title, dcc.Graph(figure=figure, config={"displayModeBar": False}), subtitle)


def empty_state(title, message):
    return dbc.Alert(
        [html.Strong(title), html.Div(message, className="mt-1")],
        color="light",
        className="empty-state",
    )


def format_money(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f} FCFA".replace(",", " ")


# --- HOME ------------------------------------------------------------------
def layout_home():
    products = available_products()
    regions = sorted(df_full["Region_Vente"].dropna().unique()) if not df_full.empty else []
    min_date, max_date = date_bounds()

    return html.Div(
        [
            top_nav("home"),
            dbc.Container(
                fluid=True,
                className="main-shell",
                children=[
                    html.Div(
                        className="hero-panel",
                        children=[
                            html.Div(
                                [
                                    html.Div("Plateforme d'aide à la décision agricole", className="eyebrow"),
                                    html.H1("Analyse PESTEL des prix agricoles au Cameroun"),
                                    html.P(
                                        "AaPROVIDIR centralise l'exploration des facteurs PESTEL, la lecture des anomalies et les projections de prix pour soutenir les décisions de production, d'approvisionnement et de marché.",
                                        className="hero-copy",
                                    ),
                                    html.Div(
                                        className="hero-actions",
                                        children=[
                                            dcc.Link("Ouvrir le dashboard descriptif", href="/descriptif", className="btn-primary-clean"),
                                            dcc.Link("Voir les prévisions", href="/previsions", className="btn-secondary-clean"),
                                            dcc.Link("Tester les modèles", href="/modeles", className="btn-secondary-clean"),
                                        ],
                                    ),
                                ],
                                className="hero-content",
                            ),
                            html.Div(
                                className="hero-metrics",
                                children=[
                                    kpi_card("Produits suivis", len(products), DS_ACCENT),
                                    kpi_card("Régions de vente", len(regions), DS_BLUE_LIGHT),
                                    kpi_card("Période", f"{min_date.year}-{max_date.year}", DS_BLUE),
                                ],
                            ),
                        ],
                    ),
                    dbc.Row(
                        className="g-4 mt-1",
                        children=[
                            dbc.Col(
                                section_card(
                                    "Dashboard descriptif",
                                    html.Ul(
                                        [
                                            html.Li("KPI de prix, volatilité et risque de choc."),
                                            html.Li("Analyse qualité, saisonnalité STL, corrélations PESTEL."),
                                            html.Li("Segmentation et conclusion narrative automatique."),
                                        ],
                                        className="clean-list",
                                    ),
                                ),
                                lg=6,
                            ),
                            dbc.Col(
                                section_card(
                                    "Prévisions & recommandations",
                                    html.Ul(
                                        [
                                            html.Li("Chargement automatique des modèles `.joblib` du dossier `models`."),
                                            html.Li("Projection multi-mois par produit et région."),
                                            html.Li("Recommandations opérationnelles selon tendance, risque et chocs."),
                                        ],
                                        className="clean-list",
                                    ),
                                ),
                                lg=6,
                            ),
                            dbc.Col(
                                section_card(
                                    "Tests et comparaison des modèles",
                                    html.Ul(
                                        [
                                            html.Li("Évaluation automatique des modèles `.joblib` chargés."),
                                            html.Li("Comparaison MAE, RMSE, MAPE et R² sur les données historiques."),
                                            html.Li("Baseline naïve pour contrôler la valeur ajoutée du modèle."),
                                        ],
                                        className="clean-list",
                                    ),
                                ),
                                lg=12,
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )


# --- DESCRIPTIVE DASHBOARD -------------------------------------------------
def layout_descriptif():
    min_date, max_date = date_bounds()
    product = default_product()

    return html.Div(
        [
            top_nav("descriptif"),
            dbc.Container(
                fluid=True,
                className="main-shell",
                children=[
                    page_header(
                        "Dashboard descriptif EDA",
                        "Lecture complète des marchés agricoles",
                        "Explorez les prix, la qualité des données, les facteurs PESTEL, les clusters et les anomalies pour une combinaison produit-région.",
                    ),
                    dbc.Card(
                        className="filter-card",
                        children=dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("PRODUIT", className="filter-label"),
                                                dcc.Dropdown(
                                                    id="filter-produit",
                                                    options=[{"label": p, "value": p} for p in available_products()],
                                                    value=product,
                                                    clearable=False,
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                            className="mb-3 mb-lg-0",
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("RÉGION DE VENTE", className="filter-label"),
                                                dcc.Dropdown(id="filter-region", options=[], clearable=False),
                                            ],
                                            lg=3,
                                            md=6,
                                            className="mb-3 mb-lg-0",
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("PLAGE TEMPORELLE", className="filter-label"),
                                                html.Div(
                                                    dcc.DatePickerRange(
                                                        id="filter-date",
                                                        min_date_allowed=min_date,
                                                        max_date_allowed=max_date,
                                                        start_date=min_date,
                                                        end_date=max_date,
                                                        display_format="MMM YYYY",
                                                    ),
                                                    className="date-wrapper",
                                                ),
                                            ],
                                            lg=6,
                                            md=12,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label("NOMBRE DE CLUSTERS KMEANS", className="filter-label mt-4"),
                                        dcc.Slider(
                                            id="cluster-count",
                                            min=2,
                                            max=min(10, max(2, len(available_products()))),
                                            step=1,
                                            value=5,
                                            marks={i: str(i) for i in range(2, min(10, max(2, len(available_products()))) + 1)},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ],
                                    className="cluster-control",
                                ),
                            ]
                        ),
                    ),
                    dcc.Tabs(
                        [
                            dcc.Tab(label="SYNTHÈSE", value="tab-metier", className="dash-tab", selected_className="dash-tab selected"),
                            dcc.Tab(label="AUDIT QUALITÉ", value="tab-audit", className="dash-tab", selected_className="dash-tab selected"),
                            dcc.Tab(label="ANALYSE PRIX", value="tab-prix", className="dash-tab", selected_className="dash-tab selected"),
                            dcc.Tab(label="FACTEURS PESTEL", value="tab-correlations", className="dash-tab", selected_className="dash-tab selected"),
                            dcc.Tab(label="SEGMENTATION", value="tab-clustering", className="dash-tab", selected_className="dash-tab selected"),
                            dcc.Tab(label="CHOCS ET ANOMALIES", value="tab-anomalies", className="dash-tab", selected_className="dash-tab selected"),
                            dcc.Tab(label="CONCLUSION", value="tab-conclusion", className="dash-tab", selected_className="dash-tab selected"),
                        ],
                        id="tabs-main",
                        value="tab-metier",
                        className="tabs-clean",
                    ),
                    dcc.Loading(id="loading-content", type="circle", children=html.Div(id="tab-content")),
                ],
            ),
        ]
    )


def filtered_df(produit, region, start, end):
    if df_full.empty or not produit or not region:
        return pd.DataFrame()
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return df_full[
        (df_full["Produit_ID"] == produit)
        & (df_full["Region_Vente"] == region)
        & (df_full["Date"] >= start_dt)
        & (df_full["Date"] <= end_dt)
    ].sort_values("Date").copy()


def layout_metier(df, produit, region):
    fig_main = px.line(
        df,
        x="Date",
        y=TARGET_COL,
        title=f"Série temporelle : {produit} ({region})",
        labels={TARGET_COL: "Prix (FCFA/kg)"},
        template="plotly_white",
        color_discrete_sequence=[DS_BLUE_LIGHT],
    )
    fig_main.update_traces(line_width=3)
    fig_main.update_layout(margin=dict(l=20, r=20, t=60, b=20))

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(kpi_card("Prix de vente actuel", format_money(df[TARGET_COL].iloc[-1])), lg=3, md=6),
                    dbc.Col(kpi_card("Moyenne historique", format_money(df[TARGET_COL].mean())), lg=3, md=6),
                    dbc.Col(kpi_card("Volatilité", format_money(df[TARGET_COL].std())), lg=3, md=6),
                    dbc.Col(kpi_card("Score de risque choc", f"{df['Score_Choc'].mean():.2f}", DS_WARNING), lg=3, md=6),
                ],
                className="g-4 mb-4",
            ),
            graph_card("Indicateur de référence", fig_main, "Trajectoire du prix de vente sur la période filtrée."),
        ]
    )


def layout_audit(df):
    null_count = df.isnull().sum()
    null_pct = (null_count / len(df) * 100).reset_index()
    null_pct.columns = ["Variable", "Pct_Null"]
    null_pct["Status"] = null_pct["Pct_Null"].apply(lambda x: "Incomplet" if x > 0 else "Complet")
    null_pct["Affichage"] = null_pct["Pct_Null"].where(null_pct["Pct_Null"] > 0, 0.4)

    fig_null = px.bar(
        null_pct,
        x="Affichage",
        y="Variable",
        orientation="h",
        title="Analyse de complétude (%)",
        labels={"Affichage": "% manquant"},
        color="Status",
        color_discrete_map={"Complet": DS_ACCENT, "Incomplet": DS_DANGER},
        text=null_pct["Pct_Null"].map(lambda value: f"{value:.1f}%"),
    )
    fig_null.update_traces(textposition="outside", cliponaxis=False)
    fig_null.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis=dict(range=[0, max(5, float(null_pct["Pct_Null"].max()) + 5)], ticksuffix="%"),
        plot_bgcolor="white",
        margin=dict(l=20, r=55, t=60, b=20),
    )

    fig_types = px.pie(
        names=["Numérique", "Catégorie", "Date"],
        values=[
            len(df.select_dtypes(include=np.number).columns),
            len(df.select_dtypes(include="object").columns),
            1,
        ],
        title="Répartition typologique",
        hole=0.45,
        color_discrete_sequence=[DS_BLUE_LIGHT, DS_ACCENT, "#cbd5e1"],
    )
    completeness_score = max(0, 100 - float(null_pct["Pct_Null"].mean()))
    duplicate_rows = int(df.duplicated().sum())
    expected_months = max(1, len(pd.period_range(df["Date"].min(), df["Date"].max(), freq="M")))
    observed_months = int(df["Date"].dt.to_period("M").nunique())
    coverage = observed_months / expected_months * 100

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(kpi_card("Score qualité", f"{completeness_score:.1f} %", DS_ACCENT), lg=3, md=6),
                    dbc.Col(kpi_card("Valeurs manquantes", int(null_count.sum()), DS_BLUE_LIGHT), lg=3, md=6),
                    dbc.Col(kpi_card("Doublons", duplicate_rows, DS_WARNING if duplicate_rows else DS_SUCCESS), lg=3, md=6),
                    dbc.Col(kpi_card("Couverture mensuelle", f"{coverage:.0f} %", DS_BLUE), lg=3, md=6),
                ],
                className="g-4 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(graph_card("Complétude par variable", fig_null, "Les variables à 0 % sont affichées avec une barre minimale pour rester visibles."), lg=7),
                    dbc.Col(graph_card("Types de variables", fig_types, "Répartition des colonnes utilisées par les analyses."), lg=5),
                ],
                className="g-4",
            ),
        ]
    )


def layout_prix(df, produit, region):
    fig_line = px.line(df, x="Date", y=TARGET_COL, title=f"Évolution des prix : {produit}", template="plotly_white")
    fig_line.update_traces(line=dict(color=DS_BLUE_LIGHT, width=2.5))
    fig_line.update_layout(margin=dict(l=20, r=20, t=60, b=20))

    content = [graph_card("Évolution des prix", fig_line, f"Prix de vente de {produit} dans la région {region}.")]

    if len(df) >= 24:
        df_ts = df.set_index("Date")[TARGET_COL].resample("MS").mean().interpolate()
        try:
            stl = STL(df_ts, period=12, robust=True).fit()
            fig_stl = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    "Série observée",
                    "Tendance à long terme",
                    "Profil saisonnier",
                    "Composante aléatoire",
                ),
            )
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=df_ts.values, name="Observé", line_color=DS_BLUE), row=1, col=1)
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=stl.trend, name="Tendance", line_color=DS_WARNING), row=2, col=1)
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=stl.seasonal, name="Saison", line_color=DS_SUCCESS), row=3, col=1)
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=stl.resid, name="Résidu", line_color=DS_DANGER), row=4, col=1)
            fig_stl.update_layout(height=800, showlegend=False, template="plotly_white")
            content.append(graph_card("Décomposition STL", fig_stl, "Décomposition mensuelle avec période saisonnière de 12 mois."))
        except Exception as exc:
            content.append(dbc.Alert(f"Analyse saisonnière non disponible : {exc}", color="warning"))

    return html.Div(content, className="stacked-content")


def get_price_correlations(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [c for c in num_cols if df[c].std() > 0]
    if TARGET_COL not in valid_cols or len(valid_cols) < 2:
        return pd.Series(dtype=float), pd.DataFrame()
    corr = df[valid_cols].corr()
    corr_prix = corr[TARGET_COL].drop(labels=[TARGET_COL], errors="ignore").dropna()
    corr_prix = corr_prix.reindex(corr_prix.abs().sort_values(ascending=False).index)
    return corr_prix, corr


def layout_correlations(df):
    corr_prix, corr = get_price_correlations(df)
    if corr.empty:
        return empty_state("Corrélations indisponibles", "Les variables numériques filtrées ne varient pas assez pour calculer une matrice fiable.")

    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Matrice de corrélation PESTEL",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )

    fig_bar = px.bar(
        x=corr_prix.index,
        y=corr_prix.values,
        title="Variables PESTEL les plus associées au prix",
        labels={"x": "Variable", "y": "Corrélation r"},
        color=corr_prix.values,
        color_continuous_scale="RdBu_r",
    )

    return dbc.Row(
        [
            dbc.Col(graph_card("Matrice PESTEL", fig_heat), lg=7),
            dbc.Col(graph_card("Influence sur le prix", fig_bar), lg=5),
        ],
        className="g-4",
    )


def cluster_profile(full_df, n_clusters=5):
    features = [
        "Prix_Vente_FCFA_kg",
        "Pertes_PostRecolte_Pct",
        "Superficie_Cultivee_ha",
        "Cout_Transport",
        "Temperature_Moy",
        "Score_Choc",
    ]
    if full_df.empty:
        return pd.DataFrame(), features
    features = [col for col in features if col in full_df.columns]
    profil = full_df.groupby("Produit_ID")[features].mean().dropna()
    if len(profil) < 2:
        return pd.DataFrame(), features

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(profil)
    n_clusters = max(2, min(int(n_clusters or 5), len(profil)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profil["Cluster"] = km.fit_predict(x_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(x_scaled)
    profil["PC1"], profil["PC2"] = coords[:, 0], coords[:, 1]
    return profil, features


def layout_clustering(full_df, n_clusters=5):
    profil, features = cluster_profile(full_df, n_clusters)
    if profil.empty:
        return empty_state("Segmentation indisponible", "Le volume de données ne permet pas de calculer les clusters.")

    fig_pca = px.scatter(
        profil.reset_index(),
        x="PC1",
        y="PC2",
        color="Cluster",
        text="Produit_ID",
        title=f"Segmentation PCA des filières - {int(n_clusters)} clusters KMeans",
        color_continuous_scale="Viridis",
    )
    fig_pca.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
    fig_pca.update_layout(plot_bgcolor="white")
    cluster_summary = (
        profil.reset_index()
        .groupby("Cluster")
        .agg(
            Produits=("Produit_ID", "count"),
            Prix_moyen=(TARGET_COL, "mean"),
            Risque_choc=("Score_Choc", "mean"),
            Cout_transport=("Cout_Transport", "mean"),
        )
        .reset_index()
        .round(2)
    )

    return html.Div(
        [
            graph_card(
                "Segmentation des produits",
                fig_pca,
                f"Regroupement KMeans sur {len(features)} variables standardisées, projeté en 2 dimensions par PCA.",
            ),
            section_card(
                "Profil des clusters",
                dash_table.DataTable(
                    data=cluster_summary.to_dict("records"),
                    columns=[{"name": col.replace("_", " "), "id": col} for col in cluster_summary.columns],
                    page_size=10,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Inter", "padding": "10px", "textAlign": "left"},
                    style_header={"fontWeight": "800", "backgroundColor": "#f8fafc"},
                ),
                "Synthèse métier des groupes calculés avec le nombre de clusters choisi.",
            ),
        ],
        className="stacked-content",
    )


def detect_anomalies(df):
    if df.empty or df[TARGET_COL].std() <= 0:
        return pd.DataFrame()
    work = df.copy()
    work["Z_Score"] = np.abs(stats.zscore(work[TARGET_COL]))
    return work[work["Z_Score"] > 2.0]


def layout_anomalies(df):
    anomalies = detect_anomalies(df)
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=df["Date"], y=df[TARGET_COL], name="Série normale", line_color=DS_BLUE_LIGHT))

    if not anomalies.empty:
        fig_anom.add_trace(
            go.Scatter(
                x=anomalies["Date"],
                y=anomalies[TARGET_COL],
                mode="markers",
                name="Anomalie statistique",
                marker=dict(color=DS_DANGER, size=10, symbol="circle"),
            )
        )
    fig_anom.update_layout(title="Détection des écarts de prix", template="plotly_white")

    choc_cols = [c for c in df.columns if c.startswith("Choc_")]
    fig_chocs = go.Figure()
    for idx, choc in enumerate(choc_cols):
        sub = df[df[choc] == 1]
        if not sub.empty:
            fig_chocs.add_trace(
                go.Scatter(
                    x=sub["Date"],
                    y=[idx] * len(sub),
                    mode="markers",
                    name=choc.replace("Choc_", ""),
                    marker=dict(symbol="square", size=8),
                )
            )
    fig_chocs.update_layout(
        title="Chronologie des événements PESTEL",
        height=350,
        template="plotly_white",
        yaxis=dict(tickvals=list(range(len(choc_cols))), ticktext=[c.replace("Choc_", "") for c in choc_cols]),
    )

    return html.Div(
        [
            graph_card("Anomalies de prix", fig_anom, "Points dont l'écart statistique dépasse deux écarts-types."),
            graph_card("Chocs encodés", fig_chocs, "Lecture temporelle des chocs PESTEL présents dans les données."),
        ],
        className="stacked-content",
    )


def cluster_description(produit, n_clusters=5):
    profil, features = cluster_profile(df_full, n_clusters)
    if profil.empty or produit not in profil.index:
        return "N/A", "profil non disponible"

    cluster_id = int(profil.loc[produit, "Cluster"])
    cluster_rows = profil[profil["Cluster"] == cluster_id]
    global_avg = profil[features].mean()
    cluster_avg = cluster_rows[features].mean()

    descriptors = []
    if TARGET_COL in features and cluster_avg[TARGET_COL] >= global_avg[TARGET_COL]:
        descriptors.append("prix moyen élevé")
    else:
        descriptors.append("prix moyen modéré")
    if "Score_Choc" in features and cluster_avg["Score_Choc"] >= global_avg["Score_Choc"]:
        descriptors.append("exposition aux chocs supérieure")
    if "Cout_Transport" in features and cluster_avg["Cout_Transport"] >= global_avg["Cout_Transport"]:
        descriptors.append("coûts logistiques élevés")
    if len(descriptors) == 1:
        descriptors.append("profil relativement stable")

    return cluster_id, ", ".join(descriptors)


def stl_status(df):
    if len(df) < 24:
        return "une saisonnalité STL non confirmée faute d'historique suffisant"
    try:
        df_ts = df.set_index("Date")[TARGET_COL].resample("MS").mean().interpolate()
        STL(df_ts, period=12, robust=True).fit()
        return "une saisonnalité de période 12 mois confirmée par STL"
    except Exception:
        return "une saisonnalité STL non confirmée sur la fenêtre filtrée"


def layout_conclusion(df, produit, region, cluster_count=5):
    corr_prix, _ = get_price_correlations(df)
    if corr_prix.empty:
        top_corr = "aucune variable exploitable"
        r_val = np.nan
        top_features = "N/A"
    else:
        top_corr = corr_prix.index[0]
        r_val = corr_prix.iloc[0]
        top_features = ", ".join(corr_prix.head(5).index)

    annees = [df["Date"].min().year, df["Date"].max().year]
    anomalies = detect_anomalies(df)
    choc_cols = [c for c in df.columns if c.startswith("Choc_")]
    n_choc = int((anomalies[choc_cols].sum(axis=1) > 0).sum()) if not anomalies.empty and choc_cols else 0
    cluster_id, cluster_desc = cluster_description(produit, cluster_count)
    corr_text = "N/A" if pd.isna(r_val) else f"{r_val:.2f}"
    price_start = float(df[TARGET_COL].iloc[0])
    price_end = float(df[TARGET_COL].iloc[-1])
    trend_pct = ((price_end - price_start) / price_start * 100) if price_start else 0
    recent_shock = float(df.tail(6)["Score_Choc"].mean()) if "Score_Choc" in df else 0
    volatility_ratio = float(df[TARGET_COL].std() / df[TARGET_COL].mean() * 100) if df[TARGET_COL].mean() else 0

    if abs(trend_pct) < 5:
        trend_label = "stabilité relative"
    elif trend_pct > 0:
        trend_label = "pression haussière"
    else:
        trend_label = "détente des prix"

    risk_score = 0
    risk_score += 1 if abs(trend_pct) >= 25 else 0
    risk_score += 1 if volatility_ratio >= 30 else 0
    risk_score += 1 if recent_shock >= 1 else 0
    risk_score += 1 if len(anomalies) >= max(3, len(df) * 0.04) else 0
    risk_label = ["Faible", "Modéré", "Élevé", "Critique", "Critique"][risk_score]
    risk_color = [DS_SUCCESS, DS_ACCENT, DS_WARNING, DS_DANGER, DS_DANGER][risk_score]

    bullets = [
        f"Une corrélation de {corr_text} avec {top_corr} (variable la plus influente).",
        f"{stl_status(df).capitalize()}.",
        f"{len(anomalies)} anomalies détectées, dont {n_choc} liées aux chocs encodés.",
        f"Une appartenance au cluster {cluster_id} ({cluster_desc}).",
    ]
    actions = [
        f"Prioriser le suivi de {top_corr} et des variables proches dans les prochains modèles.",
        "Conserver la saisonnalité mensuelle dans les features, car elle structure la dynamique observée.",
        "Contrôler les observations anormales avant entraînement pour éviter de sur-apprendre les chocs extrêmes.",
    ]
    if risk_score >= 2:
        actions.insert(0, "Mettre cette combinaison produit-région sous surveillance renforcée.")
    else:
        actions.insert(0, "Maintenir une veille standard avec suivi mensuel des indicateurs clés.")

    return html.Div(
        [
            dbc.Row(
                className="g-4 mb-4",
                children=[
                    dbc.Col(kpi_card("Diagnostic", trend_label, DS_BLUE_LIGHT, f"{trend_pct:+.1f} % sur la période"), lg=3, md=6),
                    dbc.Col(kpi_card("Risque analytique", risk_label, risk_color, f"Volatilité {volatility_ratio:.1f} %"), lg=3, md=6),
                    dbc.Col(kpi_card("Variable dominante", top_corr, DS_ACCENT, f"r = {corr_text}"), lg=3, md=6),
                    dbc.Col(kpi_card("Cluster", cluster_id, DS_BLUE, cluster_desc), lg=3, md=6),
                ],
            ),
            section_card(
                "Conclusion décisionnelle automatique",
                html.Div(
                    [
                        html.P(
                            f"Sur la période {annees[0]}-{annees[1]}, le prix de la {produit} dans la région {region} présente :",
                            className="conclusion-intro",
                        ),
                        dbc.Row(
                            className="g-4",
                            children=[
                                dbc.Col(
                                    [
                                        html.Div("Signaux clés", className="mini-heading"),
                                        html.Ul([html.Li(item) for item in bullets], className="conclusion-list"),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Actions recommandées", className="mini-heading"),
                                        html.Ul([html.Li(item) for item in actions], className="conclusion-list"),
                                    ],
                                    lg=6,
                                ),
                            ],
                        ),
                        html.Div(f"Features prioritaires recommandées : {top_features}.", className="feature-callout"),
                    ],
                    className="conclusion-box",
                ),
                "Synthèse générée à partir des corrélations, de la saisonnalité, des anomalies, des chocs et du clustering.",
            ),
            dbc.Row(
                className="g-4 mt-1",
                children=[
                    dbc.Col(kpi_card("Prix initial", format_money(price_start)), lg=4),
                    dbc.Col(kpi_card("Prix final", format_money(price_end), DS_ACCENT), lg=4),
                    dbc.Col(kpi_card("Anomalies", len(anomalies), DS_DANGER, f"{n_choc} avec choc encodé"), lg=4),
                ],
            ),
        ]
    )


# --- FORECAST DASHBOARD ----------------------------------------------------
def model_files():
    if not MODELS_DIR.exists():
        return []
    return sorted(
        path
        for path in MODELS_DIR.glob("*.joblib")
        if not path.name.endswith("_preprocessor.joblib")
    )


def model_options():
    return [{"label": path.name, "value": path.name} for path in model_files()]


def model_metric_records():
    records = {}
    if MODEL_METRICS_PATH.exists():
        try:
            metrics_df = pd.read_csv(MODEL_METRICS_PATH)
            for _, row in metrics_df.iterrows():
                path = row.get("path")
                if isinstance(path, str) and path.endswith(".joblib"):
                    records[Path(path).name] = row.to_dict()
        except Exception:
            records = {}

    if BEST_MODEL_METADATA_PATH.exists():
        try:
            metadata = json.loads(BEST_MODEL_METADATA_PATH.read_text(encoding="utf-8"))
            records["best_model.joblib"] = {
                **metadata,
                "name": f"best_model ({metadata.get('name', 'source inconnue')})",
                "path": "models/best_model.joblib",
            }
        except Exception:
            pass
    return records


def model_explainability_score(model_name, estimator_type=None):
    label = f"{model_name} {estimator_type or ''}".lower()
    if "regression" in label or "lineaire" in label or "linear" in label:
        return 1.00
    if "random_forest" in label or "random forest" in label:
        return 0.68
    if "xgboost" in label:
        return 0.58
    if "lstm" in label:
        return 0.25
    if "baseline" in label:
        return 0.90
    return 0.50


def score_models_for_balance(metric_rows):
    scored = []
    valid_rows = [
        row
        for row in metric_rows
        if row.get("Statut") == "OK" and pd.notna(row.get("MAE"))
    ]
    if not valid_rows:
        return scored

    maes = [float(row["MAE"]) for row in valid_rows]
    min_mae = min(maes)

    for row in valid_rows:
        precision_score = min(1.0, min_mae / max(float(row["MAE"]), 1e-9))
        explainability = model_explainability_score(row["Modèle"], row.get("Type"))
        balanced_score = PRECISION_WEIGHT * precision_score + EXPLAINABILITY_WEIGHT * explainability
        scored.append(
            {
                **row,
                "Score précision": round(precision_score, 3),
                "Explicabilité": round(explainability, 2),
                "Score équilibre": round(balanced_score, 3),
            }
        )
    return sorted(scored, key=lambda row: row["Score équilibre"], reverse=True)


def recommended_model_name(candidate_names=None):
    available = [path.name for path in model_files()]
    candidates = [name for name in (candidate_names or available) if name in available]
    if not candidates:
        return None

    records = model_metric_records()
    rows = []
    for name in candidates:
        record = records.get(name, {})
        mae = record.get("holdout_mae")
        if pd.isna(mae):
            mae = None
        rows.append(
            {
                "Modèle": name,
                "Statut": "OK",
                "Type": record.get("estimator_type", name),
                "MAE": float(mae) if mae is not None else 999999.0,
            }
        )

    scored = score_models_for_balance(rows)
    if scored:
        return scored[0]["Modèle"]
    return "best_model.joblib" if "best_model.joblib" in candidates else candidates[0]


def recommended_model_explanation(model_name):
    records = model_metric_records()
    record = records.get(model_name, {})
    estimator_type = record.get("estimator_type", "type non documenté")
    mae = record.get("holdout_mae")
    explainability = model_explainability_score(model_name, estimator_type)
    mae_text = "MAE non disponible" if pd.isna(mae) else f"MAE holdout {float(mae):.2f}"
    return (
        f"{model_name} est recommandé car il offre le meilleur compromis entre précision "
        f"({mae_text}) et explicabilité ({explainability:.2f}/1, {estimator_type})."
    )


def load_model(model_name):
    if not model_name:
        return None
    model_path = MODELS_DIR / model_name
    if not model_path.exists() or model_path.suffix != ".joblib":
        return None
    return joblib.load(model_path)


def forecast_regions(produit):
    if df_full.empty or not produit:
        return []
    return sorted(df_full[df_full["Produit_ID"] == produit]["Region_Vente"].dropna().unique())


def layout_previsions():
    products = available_products()
    product = default_product()
    regions = forecast_regions(product)
    models = model_options()
    recommended = recommended_model_name()

    return html.Div(
        [
            top_nav("previsions"),
            dbc.Container(
                fluid=True,
                className="main-shell",
                children=[
                    page_header(
                        "Dashboard prévisions & recommandations",
                        "Projection des prix et lecture décisionnelle",
                        "Chargez un modèle `.joblib` dans le dossier `models`, sélectionnez un produit et obtenez une trajectoire de prix accompagnée de recommandations.",
                    ),
                    dbc.Alert(
                        recommended_model_explanation(recommended) if recommended else "Déposez un modèle `.joblib` dans `models/` pour activer les prévisions.",
                        color="info",
                        className="model-recommendation-alert",
                    ),
                    dbc.Card(
                        className="filter-card forecast-filter-card",
                        children=dbc.CardBody(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("PRODUIT", className="filter-label"),
                                            dcc.Dropdown(
                                                id="forecast-produit",
                                                options=[{"label": p, "value": p} for p in products],
                                                value=product,
                                                clearable=False,
                                            ),
                                        ],
                                        lg=3,
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("RÉGION", className="filter-label"),
                                            dcc.Dropdown(
                                                id="forecast-region",
                                                options=[{"label": region, "value": region} for region in regions],
                                                value=regions[0] if regions else None,
                                                clearable=False,
                                            ),
                                        ],
                                        lg=3,
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("MODÈLE JOBLIB", className="filter-label"),
                                            dcc.Dropdown(
                                                id="forecast-model",
                                                options=models,
                                                value=recommended or (models[0]["value"] if models else None),
                                                placeholder="Déposer un fichier .joblib dans models/",
                                                clearable=True,
                                            ),
                                        ],
                                        lg=4,
                                        md=8,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("HORIZON", className="filter-label"),
                                            dcc.Dropdown(
                                                id="forecast-horizon",
                                                options=[{"label": f"{m} mois", "value": m} for m in [3, 6, 9, 12]],
                                                value=6,
                                                clearable=False,
                                            ),
                                        ],
                                        lg=2,
                                        md=4,
                                    ),
                                ],
                                className="g-3",
                            )
                        ),
                    ),
                    dcc.Loading(id="loading-forecast", type="circle", children=html.Div(id="forecast-content")),
                ],
            ),
        ]
    )


def baseline_forecast(df, horizon):
    last_date = df["Date"].max()
    last_price = float(df[TARGET_COL].iloc[-1])
    monthly = df.assign(Month=df["Date"].dt.month)
    trend = 0
    if len(df) >= 12:
        trend = (last_price - float(df[TARGET_COL].iloc[-12])) / 12

    rows = []
    for step in range(1, horizon + 1):
        future_date = last_date + pd.DateOffset(months=step)
        seasonal = monthly.loc[monthly["Month"] == future_date.month, TARGET_COL].mean()
        seasonal = last_price if pd.isna(seasonal) else float(seasonal)
        pred = max(0, 0.65 * seasonal + 0.35 * (last_price + trend * step))
        rows.append({"Date": future_date, "Prévision": pred})
    return pd.DataFrame(rows), "Projection indicative sans modèle joblib disponible."


def numeric_forecast_features(row):
    return [
        col
        for col in row.index
        if col != TARGET_COL and col not in FORECAST_HELPER_COLUMNS and pd.api.types.is_number(row[col])
    ]


def build_model_input(row, model):
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return pd.DataFrame([{feature: row.get(feature, 0) for feature in feature_names}])

    expected_count = getattr(model, "n_features_in_", None)
    default_features = numeric_forecast_features(row)
    all_numeric_features = [
        col for col in row.index if col != TARGET_COL and pd.api.types.is_number(row[col])
    ]

    if expected_count == len(default_features):
        features = default_features
    elif expected_count == len(all_numeric_features):
        features = all_numeric_features
    elif expected_count is None:
        features = default_features
    else:
        raise ValueError(
            f"schéma numérique incompatible : {len(default_features)} variables utiles "
            f"({len(all_numeric_features)} avec colonnes calculées), modèle attendu {expected_count}. "
            "Réentraîner avec noms de colonnes ou fournir un modèle aligné sur Dataset.csv."
        )

    return np.asarray([[row[col] for col in features]], dtype=float)


def model_forecast(df, horizon, model_name):
    model = load_model(model_name)
    if model is None:
        return baseline_forecast(df, horizon)

    rows = []
    last_row = df.sort_values("Date").iloc[-1].copy()
    previous_price = float(last_row[TARGET_COL])

    try:
        for step in range(1, horizon + 1):
            future_date = last_row["Date"] + pd.DateOffset(months=step)
            row = last_row.copy()
            row["Date"] = future_date
            if "Prix_T-1" in row.index:
                row["Prix_T-1"] = previous_price
            if "Saisonnalite_Sin" in row.index:
                row["Saisonnalite_Sin"] = np.sin(2 * np.pi * future_date.month / 12)
            if "Saisonnalite_Cos" in row.index:
                row["Saisonnalite_Cos"] = np.cos(2 * np.pi * future_date.month / 12)

            x_pred = build_model_input(row, model)
            pred = float(np.ravel(model.predict(x_pred))[0])
            previous_price = pred
            rows.append({"Date": future_date, "Prévision": pred})
        return pd.DataFrame(rows), f"Prévision générée avec le modèle joblib `{model_name}`."
    except Exception as exc:
        forecast_df, _ = baseline_forecast(df, horizon)
        return forecast_df, f"Modèle `{model_name}` chargé, mais prédiction impossible ({exc}). Repli indicatif affiché."


def recommendation_items(df, forecast_df):
    last_price = float(df[TARGET_COL].iloc[-1])
    final_price = float(forecast_df["Prévision"].iloc[-1])
    variation = ((final_price - last_price) / last_price) * 100 if last_price else 0
    corr_prix, _ = get_price_correlations(df)
    top_feature = corr_prix.index[0] if not corr_prix.empty else "les facteurs PESTEL disponibles"
    shock_recent = df.tail(6)["Score_Choc"].mean() if "Score_Choc" in df else 0

    if variation >= 8:
        market_action = "Anticiper une hausse : sécuriser les stocks et surveiller les coûts d'approvisionnement."
    elif variation <= -8:
        market_action = "Anticiper une détente : ajuster les volumes d'achat et éviter le surstockage."
    else:
        market_action = "Maintenir une stratégie graduelle : le prix projeté reste proche du niveau actuel."

    risk_action = (
        "Renforcer la veille sur les chocs récents, car le score moyen reste élevé."
        if shock_recent >= 1
        else "Conserver une veille standard : les chocs récents restent limités sur cette fenêtre."
    )

    return [
        f"Variation attendue à l'horizon : {variation:.1f} %.",
        market_action,
        risk_action,
        f"Prioriser le suivi de {top_feature}, facteur le plus associé au prix dans l'historique filtré.",
    ]


def layout_forecast_result(produit, region, model_name, horizon):
    df = df_full[(df_full["Produit_ID"] == produit) & (df_full["Region_Vente"] == region)].sort_values("Date").copy()
    if df.empty:
        return empty_state("Prévision indisponible", "Aucune donnée historique n'est disponible pour cette combinaison produit-région.")

    forecast_df, model_message = model_forecast(df, horizon, model_name)
    history = df.tail(24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["Date"], y=history[TARGET_COL], mode="lines", name="Historique", line=dict(color=DS_BLUE_LIGHT, width=3)))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Prévision"], mode="lines+markers", name="Prévision", line=dict(color=DS_ACCENT, width=3, dash="dash")))
    fig.update_layout(
        title=f"Prévision du prix : {produit} ({region})",
        template="plotly_white",
        yaxis_title="Prix (FCFA/kg)",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    recs = recommendation_items(df, forecast_df)
    final_price = forecast_df["Prévision"].iloc[-1]
    model_count = len(model_files())

    return html.Div(
        [
            dbc.Row(
                className="g-4 mb-4",
                children=[
                    dbc.Col(kpi_card("Dernier prix observé", format_money(df[TARGET_COL].iloc[-1])), lg=4),
                    dbc.Col(kpi_card(f"Prix prévu à {horizon} mois", format_money(final_price), DS_ACCENT), lg=4),
                    dbc.Col(kpi_card("Modèles détectés", model_count, DS_BLUE_LIGHT), lg=4),
                ],
            ),
            graph_card("Trajectoire prévisionnelle", fig, model_message),
            section_card(
                "Recommandations",
                html.Ul([html.Li(item) for item in recs], className="recommendation-list"),
                "Synthèse opérationnelle fondée sur la tendance prévue, les chocs récents et les corrélations historiques.",
            ),
        ],
        className="stacked-content",
    )


# --- MODEL TESTING ---------------------------------------------------------
def layout_model_tests():
    product = default_product()
    regions = forecast_regions(product)
    models = model_files()

    return html.Div(
        [
            top_nav("modeles"),
            dbc.Container(
                fluid=True,
                className="main-shell",
                children=[
                    page_header(
                        "Tests et comparaison des modèles",
                        "Contrôle des modèles `.joblib` sur les données historiques",
                        "Évaluez les modèles chargés dans `models/` sur un jeu de test temporel et comparez-les à une baseline naïve.",
                    ),
                    dbc.Card(
                        className="filter-card",
                        children=dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("PRODUIT", className="filter-label"),
                                                dcc.Dropdown(
                                                    id="model-test-produit",
                                                    options=[{"label": p, "value": p} for p in available_products()],
                                                    value=product,
                                                    clearable=False,
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("RÉGION", className="filter-label"),
                                                dcc.Dropdown(
                                                    id="model-test-region",
                                                    options=[{"label": region, "value": region} for region in regions],
                                                    value=regions[0] if regions else None,
                                                    clearable=False,
                                                ),
                                            ],
                                            lg=3,
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("TAILLE DU TEST TEMPOREL", className="filter-label"),
                                                dcc.Dropdown(
                                                    id="model-test-ratio",
                                                    options=[
                                                        {"label": "10 %", "value": 10},
                                                        {"label": "20 %", "value": 20},
                                                        {"label": "30 %", "value": 30},
                                                        {"label": "40 %", "value": 40},
                                                    ],
                                                    value=20,
                                                    clearable=False,
                                                ),
                                            ],
                                            lg=6,
                                            md=12,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("MODÈLES À COMPARER", className="filter-label mt-4"),
                                                dcc.Checklist(
                                                    id="model-test-models",
                                                    options=model_options(),
                                                    value=[path.name for path in models],
                                                    className="model-checklist",
                                                    inputClassName="model-check-input",
                                                    labelClassName="model-check-label",
                                                ),
                                            ],
                                            lg=12,
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ),
                    dcc.Loading(id="loading-model-tests", type="circle", children=html.Div(id="model-test-content")),
                    html.Div(
                        f"{len(models)} modèle(s) `.joblib` détecté(s) dans `models/`.",
                        className="model-count-note",
                    ),
                ],
            ),
        ]
    )


def metric_summary(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if non_zero.any() else np.nan
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape,
        "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
    }


def split_train_test(df, test_ratio):
    df = df.sort_values("Date").copy()
    test_size = max(6, int(len(df) * (test_ratio or 20) / 100))
    test_size = min(test_size, max(1, len(df) - 6))
    return df.iloc[:-test_size], df.iloc[-test_size:]


def evaluate_loaded_model(df, model_name, test_ratio):
    _, test_df = split_train_test(df, test_ratio)
    model = load_model(model_name)
    predictions = []

    for _, row in test_df.iterrows():
        x_pred = build_model_input(row.copy(), model)
        predictions.append(float(np.ravel(model.predict(x_pred))[0]))

    metrics = metric_summary(test_df[TARGET_COL], predictions)
    return {
        "model": model_name,
        "status": "OK",
        "type": model_metric_records().get(model_name, {}).get("estimator_type", type(model).__name__),
        "metrics": metrics,
        "predictions": pd.DataFrame(
            {
                "Date": test_df["Date"].values,
                "Prix réel": test_df[TARGET_COL].values,
                "Prévision": predictions,
                "Modèle": model_name,
            }
        ),
    }


def evaluate_baseline(df, test_ratio):
    train_df, test_df = split_train_test(df, test_ratio)
    if "Prix_T-1" in test_df:
        predictions = test_df["Prix_T-1"].astype(float).values
    else:
        predictions = np.repeat(float(train_df[TARGET_COL].iloc[-1]), len(test_df))

    metrics = metric_summary(test_df[TARGET_COL], predictions)
    return {
        "model": "Baseline Prix_T-1",
        "status": "Référence",
        "metrics": metrics,
        "predictions": pd.DataFrame(
            {
                "Date": test_df["Date"].values,
                "Prix réel": test_df[TARGET_COL].values,
                "Prévision": predictions,
                "Modèle": "Baseline Prix_T-1",
            }
        ),
    }


def layout_model_test_results(produit, region, test_ratio, selected_models=None):
    df = df_full[(df_full["Produit_ID"] == produit) & (df_full["Region_Vente"] == region)].sort_values("Date").copy()
    if df.empty:
        return empty_state("Test indisponible", "Aucune donnée disponible pour cette combinaison produit-région.")
    if len(df) < 12:
        return empty_state("Historique insuffisant", "Il faut au moins 12 observations pour créer un jeu de test temporel fiable.")

    selected_models = selected_models or [path.name for path in model_files()]
    selected_models = [name for name in selected_models if name in [path.name for path in model_files()]]
    if not selected_models:
        return empty_state("Aucun modèle sélectionné", "Choisissez au moins un modèle `.joblib` à comparer.")

    evaluations = [evaluate_baseline(df, test_ratio)]
    error_rows = []
    for model_name in selected_models:
        try:
            evaluations.append(evaluate_loaded_model(df, model_name, test_ratio))
        except Exception as exc:
            error_rows.append({"Modèle": model_name, "Statut": f"Erreur : {exc}"})

    metric_rows = []
    prediction_frames = []
    for evaluation in evaluations:
        row = {
            "Modèle": evaluation["model"],
            "Statut": evaluation["status"],
            "Type": evaluation.get("type", evaluation["status"]),
            "MAE": round(evaluation["metrics"]["MAE"], 2),
            "RMSE": round(evaluation["metrics"]["RMSE"], 2),
            "MAPE (%)": round(evaluation["metrics"]["MAPE"], 2) if not pd.isna(evaluation["metrics"]["MAPE"]) else None,
            "R²": round(evaluation["metrics"]["R2"], 3) if not pd.isna(evaluation["metrics"]["R2"]) else None,
        }
        metric_rows.append(row)
        prediction_frames.append(evaluation["predictions"])

    scored_models = score_models_for_balance(metric_rows)
    best_model = scored_models[0]["Modèle"] if scored_models else metric_rows[0]["Modèle"]
    score_lookup = {row["Modèle"]: row for row in scored_models}
    enriched_rows = []
    for row in metric_rows:
        scored = score_lookup.get(row["Modèle"], {})
        enriched_rows.append(
            {
                **row,
                "Score précision": scored.get("Score précision"),
                "Explicabilité": scored.get("Explicabilité"),
                "Score équilibre": scored.get("Score équilibre"),
            }
        )
    metrics_df = pd.DataFrame(enriched_rows).sort_values(
        ["Score équilibre", "MAE"],
        ascending=[False, True],
        na_position="last",
    )
    best_score = metrics_df.iloc[0]["Score équilibre"] if pd.notna(metrics_df.iloc[0]["Score équilibre"]) else None
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    test_dates = predictions_df["Date"].sort_values().unique()
    actual_df = predictions_df[predictions_df["Modèle"] == "Baseline Prix_T-1"].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_df["Date"],
            y=actual_df["Prix réel"],
            mode="lines+markers",
            name="Prix réel",
            line=dict(color=DS_BLUE, width=3),
        )
    )
    for model_name, sub in predictions_df.groupby("Modèle"):
        fig.add_trace(
            go.Scatter(
                x=sub["Date"],
                y=sub["Prévision"],
                mode="lines+markers",
                name=model_name,
                line=dict(width=2, dash="dash" if model_name != best_model else "solid"),
            )
        )
    fig.update_layout(
        title=f"Comparaison sur {len(test_dates)} points de test - {produit} ({region})",
        template="plotly_white",
        yaxis_title="Prix (FCFA/kg)",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return html.Div(
        [
            dbc.Row(
                className="g-4 mb-4",
                children=[
                    dbc.Col(kpi_card("Meilleur compromis", best_model, DS_ACCENT), lg=4),
                    dbc.Col(kpi_card("Score équilibre", f"{best_score:.3f}" if best_score is not None else "N/A", DS_BLUE_LIGHT), lg=4),
                    dbc.Col(kpi_card("Points de test", len(test_dates), DS_BLUE), lg=4),
                ],
            ),
            section_card(
                "Pourquoi ce modèle ?",
                html.Div(
                    [
                        html.P(
                            "Le choix privilégie un juste milieu : 60 % de précision "
                            "(MAE normalisée sur les modèles sélectionnés) et 40 % d'explicabilité "
                            "(régression > forêts/boosting > réseaux de neurones).",
                            className="mb-2",
                        ),
                        html.Div(
                            f"{best_model} est retenu car il maximise ce score d'équilibre parmi les modèles sélectionnés.",
                            className="feature-callout",
                        ),
                    ]
                ),
                "La baseline reste visible comme référence mais n'est pas candidate au meilleur modèle prédictif.",
            ),
            graph_card("Prévisions vs prix réels", fig, "Comparaison chronologique des modèles chargés et de la baseline."),
            section_card(
                "Classement des modèles",
                dash_table.DataTable(
                    data=metrics_df.to_dict("records") + error_rows,
                    columns=[
                        {"name": col, "id": col}
                        for col in [
                            "Modèle",
                            "Statut",
                            "Type",
                            "MAE",
                            "RMSE",
                            "MAPE (%)",
                            "R²",
                            "Score précision",
                            "Explicabilité",
                            "Score équilibre",
                        ]
                    ],
                    page_size=10,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "Inter", "padding": "10px", "textAlign": "left"},
                    style_header={"fontWeight": "800", "backgroundColor": "#f8fafc"},
                ),
                "Le score équilibre combine la précision et la lisibilité métier du modèle.",
            ),
        ],
        className="stacked-content",
    )


# --- ROUTING & CALLBACKS ---------------------------------------------------
app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname in (None, "", "/"):
        return layout_home()
    if pathname.startswith("/modeles"):
        return layout_model_tests()
    if pathname.startswith("/previsions"):
        return layout_previsions()
    if pathname.startswith("/descriptif") or pathname.startswith("/dashboard"):
        return layout_descriptif()
    return html.Div(
        [
            top_nav("home"),
            dbc.Container(
                className="main-shell",
                children=empty_state("Page introuvable", "Utilisez la navigation pour revenir à l'accueil."),
            ),
        ]
    )


@app.callback(
    [Output("filter-region", "options"), Output("filter-region", "value")],
    Input("filter-produit", "value"),
)
def update_region_options(produit_selected):
    regions_dispo = forecast_regions(produit_selected)
    options = [{"label": region, "value": region} for region in regions_dispo]
    return options, regions_dispo[0] if regions_dispo else None


@app.callback(
    Output("tab-content", "children"),
    [
        Input("tabs-main", "value"),
        Input("filter-produit", "value"),
        Input("filter-region", "value"),
        Input("filter-date", "start_date"),
        Input("filter-date", "end_date"),
        Input("cluster-count", "value"),
    ],
)
def render_tab_content(active_tab, produit, region, start, end, cluster_count):
    if not produit or not region:
        return dbc.Alert("Sélectionnez un produit et une région.", color="warning")

    df = filtered_df(produit, region, start, end)
    if df.empty:
        return dbc.Alert(f"Données non disponibles pour {produit} à {region}.", color="info", className="mt-4")

    if active_tab == "tab-metier":
        return layout_metier(df, produit, region)
    if active_tab == "tab-audit":
        return layout_audit(df)
    if active_tab == "tab-prix":
        return layout_prix(df, produit, region)
    if active_tab == "tab-correlations":
        return layout_correlations(df)
    if active_tab == "tab-clustering":
        return layout_clustering(df_full, cluster_count)
    if active_tab == "tab-anomalies":
        return layout_anomalies(df)
    if active_tab == "tab-conclusion":
        return layout_conclusion(df, produit, region, cluster_count)
    return layout_metier(df, produit, region)


@app.callback(
    [Output("forecast-region", "options"), Output("forecast-region", "value")],
    Input("forecast-produit", "value"),
)
def update_forecast_region_options(produit_selected):
    regions_dispo = forecast_regions(produit_selected)
    options = [{"label": region, "value": region} for region in regions_dispo]
    return options, regions_dispo[0] if regions_dispo else None


@app.callback(
    Output("forecast-content", "children"),
    [
        Input("forecast-produit", "value"),
        Input("forecast-region", "value"),
        Input("forecast-model", "value"),
        Input("forecast-horizon", "value"),
    ],
)
def render_forecast_content(produit, region, model_name, horizon):
    if not produit or not region:
        return dbc.Alert("Sélectionnez un produit et une région.", color="warning")
    return layout_forecast_result(produit, region, model_name, horizon or 6)


@app.callback(
    [Output("model-test-region", "options"), Output("model-test-region", "value")],
    Input("model-test-produit", "value"),
)
def update_model_test_region_options(produit_selected):
    regions_dispo = forecast_regions(produit_selected)
    options = [{"label": region, "value": region} for region in regions_dispo]
    return options, regions_dispo[0] if regions_dispo else None


@app.callback(
    Output("model-test-content", "children"),
    [
        Input("model-test-produit", "value"),
        Input("model-test-region", "value"),
        Input("model-test-ratio", "value"),
        Input("model-test-models", "value"),
    ],
)
def render_model_test_content(produit, region, test_ratio, selected_models):
    if not produit or not region:
        return dbc.Alert("Sélectionnez un produit et une région.", color="warning")
    return layout_model_test_results(produit, region, test_ratio or 20, selected_models)


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="0.0.0.0")

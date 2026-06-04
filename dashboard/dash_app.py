from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
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
                            )
                        ),
                    ),
                    dbc.Tabs(
                        [
                            dbc.Tab(label="SYNTHÈSE", tab_id="tab-metier"),
                            dbc.Tab(label="AUDIT QUALITÉ", tab_id="tab-audit"),
                            dbc.Tab(label="ANALYSE PRIX", tab_id="tab-prix"),
                            dbc.Tab(label="FACTEURS PESTEL", tab_id="tab-correlations"),
                            dbc.Tab(label="SEGMENTATION", tab_id="tab-clustering"),
                            dbc.Tab(label="CHOCS ET ANOMALIES", tab_id="tab-anomalies"),
                            dbc.Tab(label="CONCLUSION", tab_id="tab-conclusion"),
                        ],
                        id="tabs-main",
                        active_tab="tab-metier",
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

    fig_null = px.bar(
        null_pct,
        x="Pct_Null",
        y="Variable",
        orientation="h",
        title="Analyse de complétude (%)",
        labels={"Pct_Null": "% manquant"},
        color="Status",
        color_discrete_map={"Complet": DS_ACCENT, "Incomplet": DS_DANGER},
    )
    fig_null.update_layout(yaxis={"categoryorder": "total ascending"}, plot_bgcolor="white")

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

    return dbc.Row(
        [
            dbc.Col(graph_card("Complétude", fig_null), lg=7),
            dbc.Col(graph_card("Types de variables", fig_types), lg=5),
        ],
        className="g-4",
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


def cluster_profile(full_df):
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
    n_clusters = min(5, len(profil))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profil["Cluster"] = km.fit_predict(x_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(x_scaled)
    profil["PC1"], profil["PC2"] = coords[:, 0], coords[:, 1]
    return profil, features


def layout_clustering(full_df):
    profil, _ = cluster_profile(full_df)
    if profil.empty:
        return empty_state("Segmentation indisponible", "Le volume de données ne permet pas de calculer les clusters.")

    fig_pca = px.scatter(
        profil.reset_index(),
        x="PC1",
        y="PC2",
        color="Cluster",
        text="Produit_ID",
        title="Segmentation PCA des filières",
        color_continuous_scale="Viridis",
    )
    fig_pca.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
    fig_pca.update_layout(plot_bgcolor="white")
    return graph_card("Segmentation des produits", fig_pca, "Regroupement des filières selon leurs profils prix, chocs, coûts et production.")


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


def cluster_description(produit):
    profil, features = cluster_profile(df_full)
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


def layout_conclusion(df, produit, region):
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
    cluster_id, cluster_desc = cluster_description(produit)
    corr_text = "N/A" if pd.isna(r_val) else f"{r_val:.2f}"

    bullets = [
        f"Une corrélation de {corr_text} avec {top_corr} (variable la plus influente).",
        f"{stl_status(df).capitalize()}.",
        f"{len(anomalies)} anomalies détectées, dont {n_choc} liées aux chocs encodés.",
        f"Une appartenance au cluster {cluster_id} ({cluster_desc}).",
    ]

    return html.Div(
        [
            section_card(
                "Conclusion automatique",
                html.Div(
                    [
                        html.P(
                            f"Sur la période {annees[0]}-{annees[1]}, le prix de la {produit} dans la région {region} présente :",
                            className="conclusion-intro",
                        ),
                        html.Ul([html.Li(item) for item in bullets], className="conclusion-list"),
                        html.Div(f"Features prioritaires recommandées : {top_features}.", className="feature-callout"),
                    ],
                    className="conclusion-box",
                ),
                "Résumé narratif généré à partir des corrélations, de la saisonnalité, des anomalies et de la segmentation.",
            ),
            dbc.Row(
                className="g-4 mt-1",
                children=[
                    dbc.Col(kpi_card("Variable dominante", top_corr, DS_BLUE_LIGHT), lg=4),
                    dbc.Col(kpi_card("Anomalies", len(anomalies), DS_DANGER), lg=4),
                    dbc.Col(kpi_card("Cluster", cluster_id, DS_ACCENT, cluster_desc), lg=4),
                ],
            ),
        ]
    )


# --- FORECAST DASHBOARD ----------------------------------------------------
def model_files():
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.joblib"))


def model_options():
    return [{"label": path.name, "value": path.name} for path in model_files()]


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
    models = model_options()

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
                    dbc.Card(
                        className="filter-card",
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
                                            dcc.Dropdown(id="forecast-region", options=[], clearable=False),
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
                                                value=models[0]["value"] if models else None,
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


def model_forecast(df, horizon, model_name):
    model = load_model(model_name)
    if model is None:
        return baseline_forecast(df, horizon)

    rows = []
    last_row = df.sort_values("Date").iloc[-1].copy()
    previous_price = float(last_row[TARGET_COL])
    feature_names = getattr(model, "feature_names_in_", None)

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

            if feature_names is not None:
                x_pred = pd.DataFrame([{feature: row.get(feature, 0) for feature in feature_names}])
            else:
                numeric_cols = [
                    col
                    for col in row.index
                    if col != TARGET_COL and pd.api.types.is_number(row[col])
                ]
                x_pred = pd.DataFrame([{col: row[col] for col in numeric_cols}])

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


# --- ROUTING & CALLBACKS ---------------------------------------------------
app.layout = html.Div([dcc.Location(id="url", refresh=False), html.Div(id="page-content")])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname in (None, "", "/"):
        return layout_home()
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
        Input("tabs-main", "active_tab"),
        Input("filter-produit", "value"),
        Input("filter-region", "value"),
        Input("filter-date", "start_date"),
        Input("filter-date", "end_date"),
    ],
)
def render_tab_content(active_tab, produit, region, start, end):
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
        return layout_clustering(df_full)
    if active_tab == "tab-anomalies":
        return layout_anomalies(df)
    if active_tab == "tab-conclusion":
        return layout_conclusion(df, produit, region)
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


if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="0.0.0.0")

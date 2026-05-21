import pandas as pd
import numpy as np
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io

# ─── CONFIGURATION & THÈME ──────────────────────────────────────────────────
DS_BLUE = "#0056b3"
DS_ACCENT = "#00b0ff"
DS_BG = "#f4f7fa"

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.FLATLY], 
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'Dataset.csv'

# ─── CHARGEMENT DES DONNÉES ─────────────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
        df['Date'] = pd.to_datetime(df['Date'])
        choc_cols = [c for c in df.columns if 'Choc_' in c]
        df['Score_Choc'] = df[choc_cols].sum(axis=1)
        return df
    except Exception as e:
        print(f"Erreur chargement : {e}")
        return pd.DataFrame()

df_full = load_data()

# ─── COMPONENTS ─────────────────────────────────────────────────────────────
def kpi_card(label, value, color=DS_BLUE):
    return dbc.Card([
        dbc.CardBody([
            html.Div(label, className="kpi-label mb-1 text-uppercase"),
            html.H3(f"{value}", className="kpi-value mb-0", style={'color': color})
        ])
    ], className="shadow-sm border-0 mb-4 h-100")

# ─── LAYOUT ─────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dbc.NavbarSimple(
        brand="AaPROVIDIR — PLATEFORME ANALYTIQUE",
        brand_href="#",
        color=DS_BLUE,
        dark=True,
        className="mb-4 shadow-sm",
        fluid=True
    ),

    dbc.Container([
        # Zone des Filtres
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("PRODUIT", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id='filter-produit',
                            options=[{'label': p, 'value': p} for p in sorted(df_full['Produit_ID'].unique())],
                            value='Maïs',
                            clearable=False,
                        )
                    ], lg=3, md=6, className="mb-3 mb-lg-0"),
                    dbc.Col([
                        html.Label("RÉGION DE VENTE", className="small fw-bold mb-1"),
                        dcc.Dropdown(
                            id='filter-region',
                            options=[], 
                            clearable=False,
                        )
                    ], lg=3, md=6, className="mb-3 mb-lg-0"),
                    dbc.Col([
                        html.Label("PLAGE TEMPORELLE", className="small fw-bold mb-1"),
                        html.Div([
                            dcc.DatePickerRange(
                                id='filter-date',
                                min_date_allowed=df_full['Date'].min(),
                                max_date_allowed=df_full['Date'].max(),
                                start_date=df_full['Date'].min(),
                                end_date=df_full['Date'].max(),
                                display_format='MMM YYYY',
                                style={'border': 'none'}
                            )
                        ], className="p-1 border rounded bg-white")
                    ], lg=6, md=12),
                ])
            ])
        ], className="mb-4 border-0 shadow-sm"),

        # Onglets
        dbc.Tabs([
            dbc.Tab(label="SYNTHÈSE", tab_id="tab-metier"),
            dbc.Tab(label="AUDIT QUALITÉ", tab_id="tab-audit"),
            dbc.Tab(label="ANALYSE PRIX", tab_id="tab-prix"),
            dbc.Tab(label="FACTEURS PESTEL", tab_id="tab-correlations"),
            dbc.Tab(label="SEGMENTATION", tab_id="tab-clustering"),
            dbc.Tab(label="CHOCS ET ANOMALIES", tab_id="tab-anomalies"),
        ], id="tabs-main", active_tab="tab-metier", className="mb-4 nav-fill"),

        dcc.Loading(
            id="loading-content",
            type="circle",
            children=html.Div(id="tab-content")
        )
    ], fluid=True, style={'minHeight': '100vh', 'paddingBottom': '50px'})
])

@app.callback(
    [Output('filter-region', 'options'),
     Output('filter-region', 'value')],
    [Input('filter-produit', 'value')]
)
def update_region_options(produit_selected):
    regions_dispo = sorted(df_full[df_full['Produit_ID'] == produit_selected]['Region_Vente'].unique())
    options = [{'label': r, 'value': r} for r in regions_dispo]
    default_val = regions_dispo[0] if regions_dispo else None
    return options, default_val

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs-main", "active_tab"),
     Input("filter-produit", "value"),
     Input("filter-region", "value"),
     Input("filter-date", "start_date"),
     Input("filter-date", "end_date")]
)
def render_tab_content(active_tab, produit, region, start, end):
    if not produit or not region:
        return dbc.Alert("Sélectionnez un produit et une région.", color="warning")

    df = df_full[(df_full['Produit_ID'] == produit) & 
                 (df_full['Region_Vente'] == region) &
                 (df_full['Date'] >= start) & 
                 (df_full['Date'] <= end)].sort_values('Date').copy()

    if df.empty:
        return dbc.Alert(f"Données non disponibles pour {produit} à {region}.", color="info", className="mt-4")

    if active_tab == "tab-metier":
        return layout_metier(df, produit, region)
    elif active_tab == "tab-audit":
        return layout_audit(df)
    elif active_tab == "tab-prix":
        return layout_prix(df, produit, region)
    elif active_tab == "tab-correlations":
        return layout_correlations(df)
    elif active_tab == "tab-clustering":
        return layout_clustering(df_full)
    elif active_tab == "tab-anomalies":
        return layout_anomalies(df)

def layout_metier(df, produit, region):
    fig_main = px.line(df, x='Date', y='Prix_Vente_FCFA_kg', 
                       title=f"Série Temporelle : {produit} ({region})",
                       labels={'Prix_Vente_FCFA_kg': 'Prix (FCFA/kg)'},
                       template="plotly_white", color_discrete_sequence=[DS_BLUE])
    fig_main.update_traces(line_width=3)

    return html.Div([
        dbc.Row([
            dbc.Col(kpi_card("Prix de Vente Actuel", f"{df['Prix_Vente_FCFA_kg'].iloc[-1]} F"), lg=3, md=6),
            dbc.Col(kpi_card("Moyenne Historique", f"{df['Prix_Vente_FCFA_kg'].mean():.0f} F"), lg=3, md=6),
            dbc.Col(kpi_card("Volatilité (E-T)", f"{df['Prix_Vente_FCFA_kg'].std():.1f} F"), lg=3, md=6),
            dbc.Col(kpi_card("Score de Risque Choc", f"{df['Score_Choc'].mean():.2f}"), lg=3, md=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Indicateur de Référence", className="bg-white fw-bold"),
                    dbc.CardBody(dcc.Graph(figure=fig_main, config={'displayModeBar': False}))
                ], className="border-0 shadow-sm")
            ], md=12)
        ])
    ])

def layout_audit(df):
    null_count = df.isnull().sum()
    null_pct = (null_count / len(df) * 100).reset_index()
    null_pct.columns = ['Variable', 'Pct_Null']
    null_pct['Status'] = null_pct['Pct_Null'].apply(lambda x: 'Incomplet' if x > 0 else 'Complet')
    
    fig_null = px.bar(null_pct, x='Pct_Null', y='Variable', orientation='h', 
                      title="Analyse de Complétude (%)",
                      labels={'Pct_Null': '% Manquant'},
                      color='Status',
                      color_discrete_map={'Complet': DS_BLUE, 'Incomplet': '#e74c3c'})
    fig_null.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='white')

    fig_types = px.pie(names=['Numérique', 'Catégorie', 'Date'], 
                       values=[len(df.select_dtypes(include=np.number).columns),
                               len(df.select_dtypes(include='object').columns),
                               1],
                       title="Répartition Typologique",
                       hole=0.4, color_discrete_sequence=[DS_BLUE, DS_ACCENT, "#cbd5e0"])

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_null), md=7),
            dbc.Col(dcc.Graph(figure=fig_types), md=5),
        ])
    ])

def layout_prix(df, produit, region):
    fig_line = px.line(df, x='Date', y='Prix_Vente_FCFA_kg', 
                       title=f"Évolution des Prix : {produit}",
                       template="plotly_white")
    fig_line.update_traces(line=dict(color=DS_BLUE, width=2.5))

    content = [dbc.Row([dbc.Col(dcc.Graph(figure=fig_line), md=12)], className="mb-4")]
    
    if len(df) >= 24:
        df_ts = df.set_index('Date')['Prix_Vente_FCFA_kg'].resample('MS').mean().interpolate()
        try:
            stl = STL(df_ts, period=12, robust=True).fit()
            fig_stl = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=("Série Observée", "Tendance à Long Terme", 
                                                "Profil Saisonnier", "Composante Aléatoire (Résidus)"))
            
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=df_ts.values, name="Obs", line_color=DS_BLUE), row=1, col=1)
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=stl.trend, name="Trend", line_color="#e67e22"), row=2, col=1)
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=stl.seasonal, name="Season", line_color="#27ae60"), row=3, col=1)
            fig_stl.add_trace(go.Scatter(x=df_ts.index, y=stl.resid, name="Resid", line_color="#c0392b"), row=4, col=1)
            fig_stl.update_layout(height=800, showlegend=False, template="plotly_white")
            content.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_stl), md=12)]))
        except Exception as e:
            content.append(dbc.Alert(f"Analyse saisonnière non disponible : {e}", color="warning"))

    return html.Div(content)

def layout_correlations(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [c for c in num_cols if df[c].std() > 0]
    corr = df[valid_cols].corr()
    
    fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto",
                         title="Matrice de Corrélation PESTEL",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    
    target = 'Prix_Vente_FCFA_kg'
    if target in corr.columns:
        top_corr = corr[target].sort_values(ascending=False).drop(target)
        fig_bar = px.bar(x=top_corr.index, y=top_corr.values, 
                        title="Impact des Variables PESTEL sur le Prix",
                        labels={'x': 'Variable', 'y': 'Corrélation r'},
                        color=top_corr.values, color_continuous_scale="RdBu_r")
    else:
        fig_bar = go.Figure()

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_heat), md=7),
            dbc.Col(dcc.Graph(figure=fig_bar), md=5),
        ])
    ])

def layout_clustering(df_full):
    features = ['Prix_Vente_FCFA_kg', 'Pertes_PostRecolte_Pct', 'Superficie_Cultivee_ha', 
                'Cout_Transport', 'Temperature_Moy', 'Score_Choc']
    
    profil = df_full.groupby('Produit_ID')[features].mean().dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profil)
    
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    profil['Cluster'] = km.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    profil['PC1'], profil['PC2'] = coords[:,0], coords[:,1]
    
    fig_pca = px.scatter(profil.reset_index(), x='PC1', y='PC2', color='Cluster', 
                         text='Produit_ID', title="Segmentation PCA des Filières",
                         color_continuous_scale="Viridis")
    fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='white')))
    fig_pca.update_layout(plot_bgcolor='white')

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_pca), md=12),
        ])
    ])

def layout_anomalies(df):
    std = df['Prix_Vente_FCFA_kg'].std()
    if std > 0:
        df['Z_Score'] = np.abs(stats.zscore(df['Prix_Vente_FCFA_kg']))
        anomalies = df[df['Z_Score'] > 2.0]
    else:
        anomalies = pd.DataFrame()
    
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(x=df['Date'], y=df['Prix_Vente_FCFA_kg'], name="Série Normale", line_color=DS_BLUE))
    
    if not anomalies.empty:
        fig_anom.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Prix_Vente_FCFA_kg'], 
                                    mode='markers', name='Anomalie Statistique',
                                    marker=dict(color='#e74c3c', size=10, symbol='circle')))
    
    fig_anom.update_layout(title="Détection des Écarts de Prix", template="plotly_white")

    # Timeline des chocs PESTEL
    choc_cols = [c for c in df.columns if 'Choc_' in c]
    fig_chocs = go.Figure()
    for i, choc in enumerate(choc_cols):
        sub = df[df[choc] == 1]
        if not sub.empty:
            fig_chocs.add_trace(go.Scatter(x=sub['Date'], y=[i]*len(sub), mode='markers', 
                                        name=choc.replace('Choc_', ''),
                                        marker=dict(symbol='square', size=8)))
    
    fig_chocs.update_layout(title="Chronologie des Événements PESTEL", height=350, template="plotly_white",
                            yaxis=dict(tickvals=list(range(len(choc_cols))), 
                                       ticktext=[c.replace('Choc_', '') for c in choc_cols]))

    return html.Div([
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_anom), md=12)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_chocs), md=12)])
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')

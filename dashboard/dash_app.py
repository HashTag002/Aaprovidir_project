"""
PESTEL Agricultural Price Dashboard
Module 1 : Dashboard interactif — visualisation données d'entrée
"""

import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'pestel_agricole.csv'

PESTEL_COLORS = {
    'Politique':      '#E8A838',
    'Économique':     '#4ECDC4',
    'Social':         '#A78BFA',
    'Technologique':  '#34D399',
    'Environnemental':'#6EE7B7',
    'Légal':          '#F87171',
}

PESTEL_COLS = {
    'Politique':      ['stabilite_politique', 'politique_commerciale', 'subventions_agricoles'],
    'Économique':     ['pib_mondial_growth', 'inflation_us', 'taux_change_usd_eur', 'demande_mondiale'],
    'Social':         ['croissance_population', 'urbanisation_pct', 'revenu_moyen'],
    'Technologique':  ['investissement_agritech', 'rendement_agricole_index', 'adoption_irrigation'],
    'Environnemental':['anomalie_temperature', 'precipitations_index', 'superficie_cultivee_mha', 'indice_secheresse'],
    'Légal':          ['reglementation_export', 'normes_phytosanitaires', 'accord_commerce_international'],
}

PRODUCT_COLORS = {
    'Cacao': '#CD853F',
    'Café':  '#A0522D',
    'Maïs':  '#DAA520',
    'Soja':  '#6B8E23',
    'Coton': '#B8C4C8',
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_default():
    df = pd.read_csv(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def df_to_json(df):
    d = df.copy()
    if 'date' in d.columns:
        d['date'] = d['date'].astype(str)
    return d.to_json(orient='split')

def json_to_df(j):
    df = pd.read_json(io.StringIO(j), orient='split')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def empty_fig():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.3)'),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text='Aucune donnée', x=0.5, y=0.5, showarrow=False,
                          font=dict(color='rgba(255,255,255,0.2)', size=14))],
    )
    return fig

def base_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
    )

def kpi_card(title, value, unit, color, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '22px', 'marginBottom': '6px'}),
        html.Div(title, style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
            'letterSpacing': '2px', 'color': 'rgba(255,255,255,0.45)',
            'textTransform': 'uppercase', 'marginBottom': '4px',
        }),
        html.Span(value, style={
            'fontFamily': 'Syne', 'fontSize': '24px', 'fontWeight': '800', 'color': color,
        }),
        html.Span(f' {unit}', style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '11px', 'color': 'rgba(255,255,255,0.35)',
        }),
    ], style={
        'background': 'rgba(255,255,255,0.04)',
        'border': '1px solid rgba(255,255,255,0.10)',
        'borderTop': f'3px solid {color}',
        'borderRadius': '12px', 'padding': '16px 20px',
        'textAlign': 'center', 'flex': '1', 'minWidth': '150px',
    })

CARD = {
    'background': 'rgba(255,255,255,0.04)',
    'border': '1px solid rgba(255,255,255,0.10)',
    'borderRadius': '12px', 'padding': '20px', 'marginBottom': '16px',
}

def section_title(text, color):
    return html.Div(text, style={
        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '13px',
        'color': color, 'letterSpacing': '1px', 'marginBottom': '12px',
    })

# ─── APP ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='PESTEL — Marchés Agricoles',
)

# Pré-charger les données par défaut pour initialiser le store
_df0 = load_default()
_dates0 = sorted(_df0['date'].unique()) if 'date' in _df0.columns else []
_n0 = len(_dates0)
_marks0 = {i: str(pd.Timestamp(d).year) for i, d in enumerate(_dates0) if i % 12 == 0 or i == _n0 - 1}
_num_cols0 = [c for c in _df0.select_dtypes(include='number').columns if c != 'prix_usd_tonne']
_opts0 = [{'label': c.replace('_', ' ').title(), 'value': c} for c in _num_cols0]

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
app.layout = html.Div(style={'background': '#0d1520', 'minHeight': '100vh',
                              'fontFamily': 'Syne, sans-serif', 'color': '#e8eaed'}, children=[

    # HEADER
    html.Div(style={
        'background': 'linear-gradient(135deg,#0f1923,#1a2533,#0f1923)',
        'borderBottom': '1px solid rgba(78,205,196,0.3)',
        'padding': '18px 32px', 'display': 'flex',
        'alignItems': 'center', 'justifyContent': 'space-between',
    }, children=[
        html.Div([
            html.Div('PESTEL ANALYTICS', style={
                'fontFamily': 'Syne', 'fontWeight': '800', 'fontSize': '20px',
                'letterSpacing': '3px', 'color': '#4ECDC4',
            }),
            html.Div("Marchés Agricoles · Visualisation des données d'entrée", style={
                'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
                'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '1px',
            }),
        ]),
        html.Div([
            html.Span('● ', style={'color': '#34D399', 'fontSize': '11px'}),
            html.Span('MODULE 1 / 3 — INPUT DASHBOARD', style={
                'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                'color': 'rgba(255,255,255,0.35)', 'letterSpacing': '2px',
            }),
        ]),
    ]),

    # CONTRÔLES
    html.Div(style={'padding': '20px 32px 0'}, children=[
        html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                        'alignItems': 'flex-end', 'marginBottom': '16px'}, children=[

            # Upload
            html.Div([
                html.Div('📂 Charger un CSV', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#4ECDC4', 'marginBottom': '6px',
                }),
                dcc.Upload(id='upload-data',
                    children=html.Div([
                        'Glisser-déposer ou ',
                        html.Span('parcourir', style={'color': '#4ECDC4', 'textDecoration': 'underline', 'cursor': 'pointer'}),
                    ], style={'fontFamily': 'JetBrains Mono', 'fontSize': '12px', 'color': 'rgba(255,255,255,0.55)'}),
                    style={
                        'border': '1px dashed rgba(78,205,196,0.4)', 'borderRadius': '8px',
                        'padding': '10px 18px', 'cursor': 'pointer',
                        'background': 'rgba(78,205,196,0.04)',
                    }, multiple=False),
                html.Div(id='upload-status', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'color': '#34D399', 'marginTop': '4px',
                }),
            ], style={'flex': '2', 'minWidth': '260px'}),

            # Produit
            html.Div([
                html.Div('🌿 Produit', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#A78BFA', 'marginBottom': '6px',
                }),
                dcc.Dropdown(id='product-selector',
                    options=[{'label': p, 'value': p} for p in ['Tous'] + list(PRODUCT_COLORS.keys())],
                    value='Tous', clearable=False,
                    style={'minWidth': '140px'},
                ),
            ], style={'flex': '1', 'minWidth': '130px'}),

            # Slider date
            html.Div([
                html.Div('📅 Période', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#E8A838', 'marginBottom': '6px',
                }),
                dcc.RangeSlider(
                    id='date-slider', min=0, max=_n0 - 1, step=1,
                    value=[0, _n0 - 1], marks=_marks0,
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
            ], style={'flex': '3', 'minWidth': '280px'}),

            # Catégorie PESTEL
            html.Div([
                html.Div('🔬 Facteur PESTEL', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#F87171', 'marginBottom': '6px',
                }),
                dcc.Dropdown(id='pestel-category',
                    options=[{'label': k, 'value': k} for k in ['Tous'] + list(PESTEL_COLS.keys())],
                    value='Tous', clearable=False,
                    style={'minWidth': '150px'},
                ),
            ], style={'flex': '1', 'minWidth': '150px'}),
        ]),
    ]),

    # KPI
    html.Div(id='kpi-section', style={'padding': '0 32px'}),

    # GRAPHIQUES
    html.Div(style={'padding': '0 32px 32px'}, children=[

        # Ligne 1 : Prix + Donut
        dbc.Row([
            dbc.Col([html.Div(style=CARD, children=[
                section_title('📈 Évolution des prix (USD/tonne)', '#4ECDC4'),
                dcc.Graph(id='price-chart', config={'displayModeBar': False}, style={'height': '300px'}),
            ])], md=8),
            dbc.Col([html.Div(style=CARD, children=[
                section_title('🥧 Répartition des produits', '#A78BFA'),
                dcc.Graph(id='donut-chart', config={'displayModeBar': False}, style={'height': '300px'}),
            ])], md=4),
        ]),

        # Ligne 2 : PESTEL time series
        dbc.Row([dbc.Col([html.Div(style=CARD, children=[
            section_title('🌐 Indicateurs PESTEL dans le temps', '#E8A838'),
            dcc.Graph(id='pestel-timeseries', config={'displayModeBar': True}, style={'height': '320px'}),
        ])], md=12)]),

        # Ligne 3 : Heatmap + Distribution
        dbc.Row([
            dbc.Col([html.Div(style=CARD, children=[
                section_title('🔥 Heatmap de corrélation PESTEL × Prix', '#34D399'),
                dcc.Graph(id='correlation-heatmap', config={'displayModeBar': False}, style={'height': '400px'}),
            ])], md=7),
            dbc.Col([html.Div(style=CARD, children=[
                section_title('📊 Distribution des indicateurs', '#F87171'),
                dcc.Dropdown(id='dist-col-selector',
                    options=_opts0, value=_num_cols0[0] if _num_cols0 else None,
                    clearable=False, style={'marginBottom': '10px'}),
                dcc.Graph(id='distribution-chart', config={'displayModeBar': False}, style={'height': '330px'}),
            ])], md=5),
        ]),

        # Ligne 4 : Scatter + Radar
        dbc.Row([
            dbc.Col([html.Div(style=CARD, children=[
                section_title('🔭 Nuage de points : Indicateur × Prix', '#4ECDC4'),
                dcc.Dropdown(id='scatter-x-col',
                    options=_opts0, value=_num_cols0[0] if _num_cols0 else None,
                    clearable=False, style={'marginBottom': '10px'}),
                dcc.Graph(id='scatter-chart', config={'displayModeBar': False}, style={'height': '310px'}),
            ])], md=7),
            dbc.Col([html.Div(style=CARD, children=[
                section_title('🕸 Radar PESTEL par produit', '#A78BFA'),
                dcc.Graph(id='radar-chart', config={'displayModeBar': False}, style={'height': '350px'}),
            ])], md=5),
        ]),

        # Ligne 5 : Tableau
        html.Div(style=CARD, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                            'alignItems': 'center', 'marginBottom': '12px'}, children=[
                section_title('📋 Données brutes', '#E8A838'),
                html.Div(id='data-info', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
                    'color': 'rgba(255,255,255,0.35)',
                }),
            ]),
            html.Div(id='data-table-container'),
        ]),
    ]),

    # Stores — initialisés avec les données par défaut
    dcc.Store(id='filtered-data-store', data=df_to_json(_df0)),
    dcc.Store(id='raw-data-store', data=None),
])


# ─── CALLBACKS ────────────────────────────────────────────────────────────────

# 1. Upload → raw-data-store
@app.callback(
    Output('raw-data-store', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        return dash.no_update, ''
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if df.empty:
            return dash.no_update, '⚠ Fichier vide'
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df_to_json(df), f'✓ {filename} — {len(df)} lignes · {len(df.columns)} colonnes'
    except Exception as e:
        return dash.no_update, f'⚠ Erreur : {e}'


# 2. Filtres → filtered-data-store + màj dropdowns + slider
@app.callback(
    Output('filtered-data-store', 'data'),
    Output('dist-col-selector', 'options'),
    Output('dist-col-selector', 'value'),
    Output('scatter-x-col', 'options'),
    Output('scatter-x-col', 'value'),
    Output('date-slider', 'max'),
    Output('date-slider', 'marks'),
    Input('raw-data-store', 'data'),
    Input('product-selector', 'value'),
    Input('date-slider', 'value'),
    Input('pestel-category', 'value'),
)
def filter_data(raw_json, product, date_range, pestel_cat):
    df = json_to_df(raw_json) if raw_json else load_default()

    # Dates disponibles
    if 'date' in df.columns:
        dates_sorted = sorted(df['date'].dropna().unique())
    else:
        dates_sorted = []

    n = len(dates_sorted)
    marks = {}
    if n > 0:
        for i, d in enumerate(dates_sorted):
            if i % 12 == 0 or i == n - 1:
                marks[i] = str(pd.Timestamp(d).year)
        slider_max = n - 1
    else:
        slider_max = 0
        marks = {0: '0'}

    # Filtrer par produit
    if product and product != 'Tous' and 'produit' in df.columns:
        df = df[df['produit'] == product]

    # Filtrer par date
    if n > 0 and 'date' in df.columns and date_range:
        s_idx = max(0, min(date_range[0], n - 1))
        e_idx = max(0, min(date_range[1], n - 1))
        df = df[(df['date'] >= dates_sorted[s_idx]) & (df['date'] <= dates_sorted[e_idx])]

    # Colonnes numériques
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != 'prix_usd_tonne']

    if pestel_cat and pestel_cat != 'Tous' and pestel_cat in PESTEL_COLS:
        filtered_cols = [c for c in PESTEL_COLS[pestel_cat] if c in df.columns]
        num_cols = filtered_cols if filtered_cols else num_cols

    opts = [{'label': c.replace('_', ' ').title(), 'value': c} for c in num_cols]
    default_col = num_cols[0] if num_cols else None

    return df_to_json(df), opts, default_col, opts, default_col, slider_max, marks


# 3. KPI Cards
@app.callback(Output('kpi-section', 'children'), Input('filtered-data-store', 'data'))
def update_kpis(json_data):
    if not json_data:
        return []
    try:
        df = json_to_df(json_data)
        cards = []
        if 'prix_usd_tonne' in df.columns:
            cards.append(kpi_card('Prix Moyen', f"{df['prix_usd_tonne'].mean():,.0f}", 'USD/T', '#4ECDC4', '💰'))
        if 'inflation_us' in df.columns:
            cards.append(kpi_card('Inflation US', f"{df['inflation_us'].mean():.1f}", '%', '#E8A838', '📉'))
        if 'pib_mondial_growth' in df.columns:
            cards.append(kpi_card('PIB Mondial', f"{df['pib_mondial_growth'].mean():.1f}", '%', '#34D399', '🌍'))
        if 'anomalie_temperature' in df.columns:
            cards.append(kpi_card('Anomalie Temp.', f"{df['anomalie_temperature'].mean():+.2f}", '°C', '#F87171', '🌡'))
        if 'date' in df.columns:
            cards.append(kpi_card('Périodes', str(df['date'].nunique()), 'mois', '#A78BFA', '📅'))
        cards.append(kpi_card('Observations', f'{len(df):,}', 'lignes', '#6EE7B7', '🗂'))
        return html.Div(children=cards,
                        style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '16px'})
    except Exception:
        return []


# 4. Prix chart
@app.callback(
    Output('price-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('product-selector', 'value'),
)
def update_price_chart(json_data, product):
    if not json_data:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if 'date' not in df.columns or 'prix_usd_tonne' not in df.columns:
            return empty_fig()
        fig = go.Figure()
        prods = ([product] if product and product != 'Tous'
                 else (list(df['produit'].unique()) if 'produit' in df.columns else [None]))
        for p in prods:
            sub = df[df['produit'] == p].sort_values('date') if p and 'produit' in df.columns else df.sort_values('date')
            color = PRODUCT_COLORS.get(p, '#4ECDC4')
            fig.add_trace(go.Scatter(
                x=sub['date'], y=sub['prix_usd_tonne'], name=str(p),
                mode='lines', line=dict(color=color, width=2),
                hovertemplate=f'<b>{p}</b><br>%{{x}}<br>%{{y:,.0f}} USD/T<extra></extra>',
            ))
        layout = base_layout()
        layout.update(dict(
            hovermode='x unified',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False, title='USD / tonne'),
        ))
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig()


# 5. Donut
@app.callback(Output('donut-chart', 'figure'), Input('filtered-data-store', 'data'))
def update_donut(json_data):
    if not json_data:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if 'produit' not in df.columns or 'prix_usd_tonne' not in df.columns:
            return empty_fig()
        agg = df.groupby('produit')['prix_usd_tonne'].mean().reset_index()
        fig = go.Figure(go.Pie(
            labels=agg['produit'], values=agg['prix_usd_tonne'], hole=0.55,
            marker_colors=[PRODUCT_COLORS.get(p, '#888') for p in agg['produit']],
            textinfo='label+percent',
            textfont=dict(family='JetBrains Mono', size=11),
            hovertemplate='<b>%{label}</b><br>Prix moyen : %{value:,.0f} USD/T<extra></extra>',
        ))
        layout = base_layout()
        layout.update(showlegend=False, annotations=[dict(
            text='Prix moy.', x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, family='Syne', color='rgba(255,255,255,0.5)'),
        )])
        fig.update_layout(**layout)
        return fig
    except Exception:
        return empty_fig()


# 6. PESTEL time series
@app.callback(
    Output('pestel-timeseries', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('pestel-category', 'value'),
    Input('product-selector', 'value'),
)
def update_pestel_ts(json_data, cat, product):
    if not json_data:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if 'date' not in df.columns:
            return empty_fig()
        if product and product != 'Tous' and 'produit' in df.columns:
            df = df[df['produit'] == product]

        if cat == 'Tous':
            cols_to_show = [(k, v[0]) for k, v in PESTEL_COLS.items() if v and v[0] in df.columns]
        else:
            cols_to_show = [(cat, c) for c in PESTEL_COLS.get(cat, []) if c in df.columns]

        if not cols_to_show:
            return empty_fig()

        num_cols = [c for _, c in cols_to_show]
        agg = df.groupby('date')[num_cols].mean().reset_index().sort_values('date')
        fig = go.Figure()
        for cat_name, col in cols_to_show:
            color = PESTEL_COLORS.get(cat_name, '#888')
            fig.add_trace(go.Scatter(
                x=agg['date'], y=agg[col],
                name=col.replace('_', ' ').title(),
                mode='lines', line=dict(color=color, width=1.8),
                hovertemplate=f'<b>{col}</b> : %{{y:.2f}}<extra></extra>',
            ))
        layout = base_layout()
        layout.update(dict(
            hovermode='x unified',
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False),
            legend=dict(bgcolor='rgba(13,21,32,0.9)', orientation='h',
                        yanchor='bottom', y=1.02, xanchor='left', x=0,
                        bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
            margin=dict(l=10, r=10, t=40, b=10),
        ))
        fig.update_layout(**layout)
        return fig
    except Exception:
        return empty_fig()


# 7. Heatmap corrélation
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('product-selector', 'value'),
)
def update_heatmap(json_data, product):
    if not json_data:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if product and product != 'Tous' and 'produit' in df.columns:
            df = df[df['produit'] == product]
        num_df = df.select_dtypes(include='number')
        if len(num_df.columns) < 2:
            return empty_fig()
        corr = num_df.corr()
        labels = [c.replace('_', '<br>') for c in corr.columns]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            colorscale=[[0, '#E8A838'], [0.5, '#1a2533'], [1, '#4ECDC4']],
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text:.1f}',
            textfont=dict(size=8, family='JetBrains Mono'),
            hovertemplate='%{y} × %{x} : %{z:.3f}<extra></extra>',
        ))
        layout = base_layout()
        layout.update(xaxis=dict(side='bottom', tickangle=-45, tickfont=dict(size=8)))
        fig.update_layout(**layout)
        return fig
    except Exception:
        return empty_fig()


# 8. Distribution
@app.callback(
    Output('distribution-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('dist-col-selector', 'value'),
    Input('product-selector', 'value'),
)
def update_distribution(json_data, col, product):
    if not json_data or not col:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if col not in df.columns:
            return empty_fig()
        if product and product != 'Tous' and 'produit' in df.columns:
            df = df[df['produit'] == product]
        cat_name = next((k for k, v in PESTEL_COLS.items() if col in v), 'Autre')
        color = PESTEL_COLORS.get(cat_name, '#4ECDC4')
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[col].dropna(), nbinsx=25,
            marker_color=color, opacity=0.8,
            hovertemplate='%{x:.2f} : %{y} obs.<extra></extra>',
        ))
        # KDE
        vals = df[col].dropna()
        if len(vals) > 5:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals)
                xr = np.linspace(vals.min(), vals.max(), 200)
                yk = kde(xr) * len(vals) * (vals.max() - vals.min()) / 25
                fig.add_trace(go.Scatter(
                    x=xr, y=yk, mode='lines',
                    line=dict(color='white', width=1.5, dash='dot'),
                    name='KDE', hoverinfo='skip',
                ))
            except Exception:
                pass
        layout = base_layout()
        layout.update(
            showlegend=False, bargap=0.05,
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=col.replace('_', ' ').title()),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Fréquence'),
        )
        fig.update_layout(**layout)
        return fig
    except Exception:
        return empty_fig()


# 9. Scatter
@app.callback(
    Output('scatter-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('scatter-x-col', 'value'),
)
def update_scatter(json_data, x_col):
    if not json_data or not x_col:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if x_col not in df.columns or 'prix_usd_tonne' not in df.columns:
            return empty_fig()
        fig = go.Figure()
        prods = list(df['produit'].unique()) if 'produit' in df.columns else [None]
        for p in prods:
            sub = df[df['produit'] == p] if p and 'produit' in df.columns else df
            color = PRODUCT_COLORS.get(p, '#4ECDC4')
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub['prix_usd_tonne'],
                mode='markers', name=str(p) if p else 'Données',
                marker=dict(color=color, size=5, opacity=0.7,
                            line=dict(width=0.5, color='rgba(255,255,255,0.15)')),
                hovertemplate=f'<b>{p}</b><br>{x_col}: %{{x:.2f}}<br>Prix: %{{y:,.0f}}<extra></extra>',
            ))
        # Trend line
        try:
            x_v = df[x_col].dropna()
            y_v = df.loc[x_v.index, 'prix_usd_tonne']
            coeffs = np.polyfit(x_v, y_v, 1)
            xr = np.linspace(x_v.min(), x_v.max(), 100)
            fig.add_trace(go.Scatter(
                x=xr, y=np.polyval(coeffs, xr),
                mode='lines', name='Tendance',
                line=dict(color='rgba(255,255,255,0.45)', width=1.5, dash='dash'),
                hoverinfo='skip',
            ))
        except Exception:
            pass
        layout = base_layout()
        layout.update(
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=x_col.replace('_', ' ').title()),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Prix USD/T'),
        )
        fig.update_layout(**layout)
        return fig
    except Exception:
        return empty_fig()


# 10. Radar
@app.callback(
    Output('radar-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('pestel-category', 'value'),
)
def update_radar(json_data, cat):
    if not json_data:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if 'produit' not in df.columns:
            return empty_fig()
        radar_cols = []
        for category, cols in PESTEL_COLS.items():
            if cat == 'Tous' or cat == category:
                radar_cols += [c for c in cols if c in df.columns]
        if not radar_cols:
            return empty_fig()
        col_min = df[radar_cols].min()
        col_max = df[radar_cols].max()
        col_range = (col_max - col_min).replace(0, 1)
        fig = go.Figure()
        for prod in df['produit'].unique():
            sub = df[df['produit'] == prod][radar_cols].mean()
            norm = ((sub - col_min) / col_range * 10).fillna(0)
            vals = list(norm.values) + [norm.values[0]]
            labels = [c.replace('_', ' ').title() for c in radar_cols] + [radar_cols[0].replace('_', ' ').title()]
            color = PRODUCT_COLORS.get(prod, '#888888')
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill='toself', name=prod,
                line=dict(color=color, width=2),
                fillcolor=color + '30',
                hovertemplate='<b>' + str(prod) + '</b><br>%{theta} : %{r:.1f}<extra></extra>',
            ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 10],
                                gridcolor='rgba(255,255,255,0.1)',
                                tickfont=dict(size=8, family='JetBrains Mono', color='rgba(255,255,255,0.35)'),
                                tickangle=0),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                                 tickfont=dict(size=9, family='JetBrains Mono', color='rgba(255,255,255,0.55)')),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)'),
            margin=dict(l=30, r=30, t=20, b=20),
        )
        return fig
    except Exception:
        return empty_fig()


# 11. Tableau
@app.callback(
    Output('data-table-container', 'children'),
    Output('data-info', 'children'),
    Input('filtered-data-store', 'data'),
)
def update_table(json_data):
    if not json_data:
        return html.Div('Aucune donnée', style={'color': 'rgba(255,255,255,0.3)',
                                                 'fontFamily': 'JetBrains Mono', 'fontSize': '12px'}), ''
    try:
        df = json_to_df(json_data)
        if 'date' in df.columns:
            df['date'] = df['date'].dt.strftime('%Y-%m').fillna('')
        preview = df.head(50)
        table = dash_table.DataTable(
            data=preview.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in preview.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgba(78,205,196,0.12)',
                'color': '#4ECDC4', 'fontFamily': 'JetBrains Mono',
                'fontSize': '10px', 'fontWeight': '600',
                'border': '1px solid rgba(78,205,196,0.2)',
                'letterSpacing': '1px', 'padding': '10px 12px',
                'textTransform': 'uppercase',
            },
            style_cell={
                'backgroundColor': 'rgba(255,255,255,0.02)',
                'color': 'rgba(255,255,255,0.75)',
                'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
                'border': '1px solid rgba(255,255,255,0.05)',
                'padding': '8px 12px', 'maxWidth': '130px',
                'overflow': 'hidden', 'textOverflow': 'ellipsis',
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(78,205,196,0.025)'},
            ],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
        )
        info = f'{len(df):,} lignes × {len(df.columns)} colonnes · 50 premières affichées'
        return table, info
    except Exception as e:
        return html.Div(f'Erreur : {e}', style={'color': '#F87171', 'fontFamily': 'JetBrains Mono'}), ''


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')

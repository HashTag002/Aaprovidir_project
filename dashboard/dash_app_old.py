"""
PESTEL Agricultural Price Dashboard
Module: dashboard interactif (visualisation données d'entrée)
Dash standalone app - intégrable dans Django via iframe ou run séparé
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import os

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'pestel_agricole.csv'

# Palette PESTEL par catégorie
PESTEL_COLORS = {
    'Politique': '#E8A838',
    'Économique': '#4ECDC4',
    'Social': '#A78BFA',
    'Technologique': '#34D399',
    'Environnemental': '#6EE7B7',
    'Légal': '#F87171',
}

PESTEL_COLS = {
    'Politique': ['stabilite_politique', 'politique_commerciale', 'subventions_agricoles'],
    'Économique': ['pib_mondial_growth', 'inflation_us', 'taux_change_usd_eur', 'demande_mondiale'],
    'Social': ['croissance_population', 'urbanisation_pct', 'revenu_moyen'],
    'Technologique': ['investissement_agritech', 'rendement_agricole_index', 'adoption_irrigation'],
    'Environnemental': ['anomalie_temperature', 'precipitations_index', 'superficie_cultivee_mha', 'indice_secheresse'],
    'Légal': ['reglementation_export', 'normes_phytosanitaires', 'accord_commerce_international'],
}

PRODUCT_COLORS = {
    'Cacao': '#8B4513',
    'Café':  '#6F4E37',
    'Maïs':  '#DAA520',
    'Soja':  '#556B2F',
    'Coton': '#B8C4C8',
}

# ─── CHARGEMENT DONNÉES ───────────────────────────────────────────────────────
def load_data(path=None):
    p = path or DATA_PATH
    df = pd.read_csv(p)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

df_global = load_data()

# ─── APP INIT ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='PESTEL — Marchés Agricoles'
)

# ─── STYLES ───────────────────────────────────────────────────────────────────
CARD_STYLE = {
    'background': 'rgba(255,255,255,0.04)',
    'border': '1px solid rgba(255,255,255,0.10)',
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '16px',
}

HEADER_STYLE = {
    'background': 'linear-gradient(135deg, #0f1923 0%, #1a2533 50%, #0f1923 100%)',
    'borderBottom': '1px solid rgba(78,205,196,0.3)',
    'padding': '20px 32px',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'space-between',
}

def kpi_card(title, value, unit, color, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '24px', 'marginBottom': '8px'}),
        html.Div(title, style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
            'letterSpacing': '2px', 'color': 'rgba(255,255,255,0.5)',
            'textTransform': 'uppercase', 'marginBottom': '4px'
        }),
        html.Span(value, style={
            'fontFamily': 'Syne', 'fontSize': '26px',
            'fontWeight': '800', 'color': color
        }),
        html.Span(f' {unit}', style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
            'color': 'rgba(255,255,255,0.4)'
        }),
    ], style={
        **CARD_STYLE,
        'textAlign': 'center',
        'borderTop': f'3px solid {color}',
        'flex': '1',
        'minWidth': '160px',
    })

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    'background': '#0d1520',
    'minHeight': '100vh',
    'fontFamily': 'Syne, sans-serif',
    'color': '#e8eaed',
}, children=[

    # ── HEADER ──
    html.Div(style=HEADER_STYLE, children=[
        html.Div([
            html.Div('PESTEL ANALYTICS', style={
                'fontFamily': 'Syne', 'fontWeight': '800',
                'fontSize': '22px', 'letterSpacing': '3px',
                'color': '#4ECDC4',
            }),
            html.Div('Marchés Agricoles · Visualisation des données d\'entrée', style={
                'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
                'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '1px',
            }),
        ]),
        html.Div([
            html.Span('● ', style={'color': '#34D399', 'fontSize': '12px'}),
            html.Span('MODULE 1 / 3 — INPUT DASHBOARD', style={
                'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '2px',
            }),
        ]),
    ]),

    # ── CONTRÔLES PRINCIPAUX ──
    html.Div(style={'padding': '24px 32px 0'}, children=[
        html.Div(style={
            'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
            'alignItems': 'flex-end', 'marginBottom': '20px'
        }, children=[

            # Upload CSV
            html.Div([
                html.Div('📂 Charger un CSV', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#4ECDC4',
                    'marginBottom': '8px', 'textTransform': 'uppercase',
                }),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        html.Span('Glisser-déposer ou '),
                        html.Span('parcourir', style={'color': '#4ECDC4', 'textDecoration': 'underline', 'cursor': 'pointer'}),
                        html.Span(' un fichier CSV')
                    ], style={'fontFamily': 'JetBrains Mono', 'fontSize': '12px', 'color': 'rgba(255,255,255,0.6)'}),
                    style={
                        'border': '1px dashed rgba(78,205,196,0.4)',
                        'borderRadius': '8px', 'padding': '12px 20px',
                        'cursor': 'pointer', 'background': 'rgba(78,205,196,0.04)',
                    },
                    multiple=False,
                ),
                html.Div(id='upload-status', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'color': '#34D399', 'marginTop': '4px'
                }),
            ], style={'flex': '2', 'minWidth': '280px'}),

            # Sélecteur produit
            html.Div([
                html.Div('🌿 Produit', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#A78BFA', 'marginBottom': '8px',
                }),
                dcc.Dropdown(
                    id='product-selector',
                    options=[{'label': p, 'value': p} for p in ['Tous'] + list(PRODUCT_COLORS.keys())],
                    value='Tous',
                    clearable=False,
                    style={'background': '#1a2533', 'border': 'none', 'color': '#fff', 'minWidth': '150px'},
                    className='dark-dropdown',
                ),
            ], style={'flex': '1', 'minWidth': '140px'}),

            # Plage temporelle
            html.Div([
                html.Div('📅 Période', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#E8A838', 'marginBottom': '8px',
                }),
                dcc.RangeSlider(
                    id='date-slider',
                    min=0, max=59, step=1,
                    value=[0, 59],
                    marks={0: '2020', 12: '2021', 24: '2022', 36: '2023', 48: '2024', 59: '2025'},
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
            ], style={'flex': '3', 'minWidth': '300px'}),

            # Catégorie PESTEL
            html.Div([
                html.Div('🔬 Facteur PESTEL', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#F87171', 'marginBottom': '8px',
                }),
                dcc.Dropdown(
                    id='pestel-category',
                    options=[{'label': k, 'value': k} for k in ['Tous'] + list(PESTEL_COLS.keys())],
                    value='Tous',
                    clearable=False,
                    style={'background': '#1a2533', 'color': '#fff', 'minWidth': '160px'},
                ),
            ], style={'flex': '1', 'minWidth': '160px'}),
        ]),
    ]),

    # ── KPI CARDS ──
    html.Div(id='kpi-section', style={'padding': '0 32px'}),

    # ── GRAPHIQUES ──
    html.Div(style={'padding': '0 32px 32px'}, children=[

        # Ligne 1 : Prix + Volume
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('📈 Évolution des Prix (USD/tonne)', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#4ECDC4', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Graph(id='price-chart', config={'displayModeBar': False}, style={'height': '320px'}),
                ])
            ], md=8),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('🥧 Répartition des données', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#A78BFA', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Graph(id='donut-chart', config={'displayModeBar': False}, style={'height': '320px'}),
                ])
            ], md=4),
        ]),

        # Ligne 2 : Indicateurs PESTEL
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('🌐 Indicateurs PESTEL dans le temps', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#E8A838', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Graph(id='pestel-timeseries', config={'displayModeBar': True}, style={'height': '340px'}),
                ])
            ], md=12),
        ]),

        # Ligne 3 : Heatmap corrélation + Distribution
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('🔥 Heatmap de Corrélation PESTEL × Prix', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#34D399', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Graph(id='correlation-heatmap', config={'displayModeBar': False}, style={'height': '420px'}),
                ])
            ], md=7),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('📊 Distributions des indicateurs', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#F87171', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Dropdown(
                        id='dist-col-selector',
                        options=[],
                        value=None,
                        clearable=False,
                        style={'background': '#1a2533', 'marginBottom': '12px'},
                    ),
                    dcc.Graph(id='distribution-chart', config={'displayModeBar': False}, style={'height': '360px'}),
                ])
            ], md=5),
        ]),

        # Ligne 4 : Scatter + Radar
        dbc.Row([
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('🔭 Nuage de points : Indicateur × Prix', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#4ECDC4', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Dropdown(
                        id='scatter-x-col',
                        options=[],
                        value=None,
                        clearable=False,
                        style={'background': '#1a2533', 'marginBottom': '12px'},
                    ),
                    dcc.Graph(id='scatter-chart', config={'displayModeBar': False}, style={'height': '340px'}),
                ])
            ], md=7),
            dbc.Col([
                html.Div(style=CARD_STYLE, children=[
                    html.Div('🕸 Radar PESTEL par Produit', style={
                        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                        'color': '#A78BFA', 'letterSpacing': '1px', 'marginBottom': '12px',
                    }),
                    dcc.Graph(id='radar-chart', config={'displayModeBar': False}, style={'height': '370px'}),
                ])
            ], md=5),
        ]),

        # Ligne 5 : Tableau de données
        html.Div(style=CARD_STYLE, children=[
            html.Div(style={
                'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'marginBottom': '12px'
            }, children=[
                html.Div('📋 Aperçu des données brutes', style={
                    'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '14px',
                    'color': '#E8A838', 'letterSpacing': '1px',
                }),
                html.Div(id='data-info', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
                    'color': 'rgba(255,255,255,0.4)',
                }),
            ]),
            html.Div(id='data-table-container'),
        ]),
    ]),

    # Store pour les données filtrées
    dcc.Store(id='filtered-data-store'),
    dcc.Store(id='raw-data-store'),
])


# ─── CALLBACKS ────────────────────────────────────────────────────────────────

# 1. Upload CSV → store
@app.callback(
    Output('raw-data-store', 'data'),
    Output('upload-status', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return dash.no_update, ''
    import base64
    import io
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df.to_json(date_format='iso', orient='split'), f'✓ {filename} — {len(df)} lignes, {len(df.columns)} colonnes'
    except Exception as e:
        return dash.no_update, f'⚠ Erreur: {str(e)}'


# 2. Filtres → données filtrées
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
    # Charger df
    if raw_json:
        df = pd.read_json(raw_json, orient='split')
    else:
        df = load_data()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        dates_sorted = sorted(df['date'].unique())
        n_dates = len(dates_sorted)
        marks = {}
        for i, d in enumerate(dates_sorted):
            if i % 12 == 0 or i == len(dates_sorted) - 1:
                marks[i] = str(pd.Timestamp(d).year)
    else:
        n_dates = len(df)
        marks = {0: '0', len(df)-1: str(len(df)-1)}

    # Filtrer produit
    if product != 'Tous' and 'produit' in df.columns:
        df = df[df['produit'] == product]

    # Filtrer dates
    if 'date' in df.columns and len(dates_sorted) > 0:
        start_idx = date_range[0] if date_range else 0
        end_idx = date_range[1] if date_range else len(dates_sorted) - 1
        start_d = dates_sorted[min(start_idx, len(dates_sorted)-1)]
        end_d = dates_sorted[min(end_idx, len(dates_sorted)-1)]
        df = df[(df['date'] >= start_d) & (df['date'] <= end_d)]

    # Colonnes numériques pour dropdowns
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != 'prix_usd_tonne']
    
    if pestel_cat != 'Tous' and pestel_cat in PESTEL_COLS:
        available = [c for c in PESTEL_COLS[pestel_cat] if c in df.columns]
        if available:
            num_cols_filtered = available
        else:
            num_cols_filtered = num_cols
    else:
        num_cols_filtered = num_cols

    opts = [{'label': c.replace('_', ' ').title(), 'value': c} for c in num_cols_filtered]
    dist_val = num_cols_filtered[0] if num_cols_filtered else None
    scatter_val = num_cols_filtered[0] if num_cols_filtered else None

    df['date'] = df['date'].astype(str)
    return df.to_json(orient='split'), opts, dist_val, opts, scatter_val, n_dates - 1, marks


# 3. KPI Cards
@app.callback(
    Output('kpi-section', 'children'),
    Input('filtered-data-store', 'data'),
)
def update_kpis(json_data):
    if not json_data:
        return []
    df = pd.read_json(json_data, orient='split')
    cards = []

    if 'prix_usd_tonne' in df.columns:
        avg_price = df['prix_usd_tonne'].mean()
        cards.append(kpi_card('Prix Moyen', f'{avg_price:,.0f}', 'USD/T', '#4ECDC4', '💰'))

    if 'inflation_us' in df.columns:
        inf = df['inflation_us'].mean()
        cards.append(kpi_card('Inflation US moy.', f'{inf:.1f}', '%', '#E8A838', '📉'))

    if 'pib_mondial_growth' in df.columns:
        pib = df['pib_mondial_growth'].mean()
        cards.append(kpi_card('Croissance PIB', f'{pib:.1f}', '%', '#34D399', '🌍'))

    if 'anomalie_temperature' in df.columns:
        tmp = df['anomalie_temperature'].mean()
        cards.append(kpi_card('Anomalie Temp.', f'{tmp:+.2f}', '°C', '#F87171', '🌡'))

    if 'date' in df.columns:
        n_periods = df['date'].nunique() if 'date' in df.columns else len(df)
        cards.append(kpi_card('Périodes', str(n_periods), 'mois', '#A78BFA', '📅'))

    n_rows = len(df)
    cards.append(kpi_card('Observations', str(n_rows), 'lignes', '#6EE7B7', '🗂'))

    return html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '16px'},
                    children=cards)


# 4. Graphique prix
@app.callback(
    Output('price-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('product-selector', 'value'),
)
def update_price_chart(json_data, product):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if 'date' not in df.columns or 'prix_usd_tonne' not in df.columns:
        return go.Figure()

    fig = go.Figure()
    products = [product] if (product and product != 'Tous') else df['produit'].unique() if 'produit' in df.columns else ['']

    for p in products:
        sub = df[df['produit'] == p] if 'produit' in df.columns and p else df
        sub = sub.sort_values('date')
        color = PRODUCT_COLORS.get(p, '#4ECDC4')
        fig.add_trace(go.Scatter(
            x=sub['date'], y=sub['prix_usd_tonne'],
            name=p, mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color.replace('#', 'rgba(').replace(')', ',0.08)') if color.startswith('#') else color,
            hovertemplate=f'<b>{p}</b><br>Date: %{{x}}<br>Prix: %{{y:,.0f}} USD/T<extra></extra>',
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=11),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, zeroline=False,
                   title='USD / tonne'),
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode='x unified',
    )
    return fig


# 5. Donut chart
@app.callback(
    Output('donut-chart', 'figure'),
    Input('filtered-data-store', 'data'),
)
def update_donut(json_data):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if 'produit' not in df.columns:
        return go.Figure()

    counts = df.groupby('produit')['prix_usd_tonne'].mean().reset_index()
    fig = go.Figure(go.Pie(
        labels=counts['produit'],
        values=counts['prix_usd_tonne'],
        hole=0.6,
        marker_colors=[PRODUCT_COLORS.get(p, '#888') for p in counts['produit']],
        textinfo='label+percent',
        textfont=dict(family='JetBrains Mono', size=11),
        hovertemplate='<b>%{label}</b><br>Prix moyen: %{value:,.0f} USD/T<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)'),
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(
            text='Prix Moy.', x=0.5, y=0.5,
            font=dict(size=13, family='Syne', color='rgba(255,255,255,0.6)'),
            showarrow=False,
        )],
    )
    return fig


# 6. PESTEL time series
@app.callback(
    Output('pestel-timeseries', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('pestel-category', 'value'),
    Input('product-selector', 'value'),
)
def update_pestel_ts(json_data, cat, product):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if 'date' not in df.columns:
        return go.Figure()

    if product != 'Tous' and 'produit' in df.columns:
        df = df[df['produit'] == product]

    if cat == 'Tous':
        # Show one representative column per category
        cols_to_show = [(k, v[0]) for k, v in PESTEL_COLS.items() if v[0] in df.columns]
    else:
        cols_to_show = [(cat, c) for c in PESTEL_COLS.get(cat, []) if c in df.columns]

    df_agg = df.groupby('date')[
        [c for _, c in cols_to_show]
    ].mean().reset_index().sort_values('date')

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    for cat_name, col in cols_to_show:
        color = PESTEL_COLORS.get(cat_name, '#888')
        fig.add_trace(go.Scatter(
            x=df_agg['date'], y=df_agg[col],
            name=col.replace('_', ' ').title(),
            mode='lines',
            line=dict(color=color, width=1.5),
            hovertemplate=f'<b>{col}</b>: %{{y:.2f}}<extra></extra>',
        ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        legend=dict(bgcolor='rgba(13,21,32,0.9)', bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1, orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified',
    )
    return fig


# 7. Heatmap corrélation
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('product-selector', 'value'),
)
def update_heatmap(json_data, product):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if product != 'Tous' and 'produit' in df.columns:
        df = df[df['produit'] == product]

    num_df = df.select_dtypes(include='number')
    if len(num_df.columns) < 2:
        return go.Figure()

    corr = num_df.corr(numeric_only=True)
    labels = [c.replace('_', '<br>') for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[
            [0, '#E8A838'], [0.5, '#1a2533'], [1, '#4ECDC4']
        ],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text:.1f}',
        textfont=dict(size=8, family='JetBrains Mono'),
        hovertemplate='%{x} × %{y}: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.6)', size=9),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(side='bottom', tickangle=-45),
    )
    return fig


# 8. Distribution
@app.callback(
    Output('distribution-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('dist-col-selector', 'value'),
    Input('product-selector', 'value'),
)
def update_distribution(json_data, col, product):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if not col or col not in df.columns:
        return go.Figure()
    if product != 'Tous' and 'produit' in df.columns:
        df = df[df['produit'] == product]

    cat_name = next((k for k, v in PESTEL_COLS.items() if col in v), 'Autre')
    color = PESTEL_COLORS.get(cat_name, '#4ECDC4')

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[col], nbinsx=25,
        marker_color=color,
        opacity=0.85,
        hovertemplate='%{x:.2f}: %{y} obs.<extra></extra>',
    ))
    # Courbe KDE approchée
    vals = df[col].dropna()
    if len(vals) > 5:
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 200)
            kde_vals = kde(x_range) * len(vals) * (vals.max() - vals.min()) / 25
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_vals,
                mode='lines', line=dict(color='white', width=1.5, dash='dot'),
                name='KDE', hoverinfo='skip',
            ))
        except Exception:
            pass

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=col.replace('_', ' ').title()),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Fréquence'),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        bargap=0.05,
    )
    return fig


# 9. Scatter
@app.callback(
    Output('scatter-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('scatter-x-col', 'value'),
)
def update_scatter(json_data, x_col):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if not x_col or x_col not in df.columns or 'prix_usd_tonne' not in df.columns:
        return go.Figure()

    cat_name = next((k for k, v in PESTEL_COLS.items() if x_col in v), 'Autre')
    color = PESTEL_COLORS.get(cat_name, '#4ECDC4')

    fig = go.Figure()
    products_list = df['produit'].unique() if 'produit' in df.columns else ['']
    for p in products_list:
        sub = df[df['produit'] == p] if 'produit' in df.columns else df
        pc = PRODUCT_COLORS.get(p, color)
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub['prix_usd_tonne'],
            mode='markers', name=p,
            marker=dict(color=pc, size=5, opacity=0.7,
                       line=dict(width=0.5, color='rgba(255,255,255,0.2)')),
            hovertemplate=f'<b>{p}</b><br>{x_col}: %{{x:.2f}}<br>Prix: %{{y:,.0f}}<extra></extra>',
        ))

    # Trend line
    try:
        from numpy.polynomial.polynomial import polyfit
        x_all = df[x_col].dropna()
        y_all = df.loc[x_all.index, 'prix_usd_tonne']
        coeffs = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=np.polyval(coeffs, x_line),
            mode='lines', name='Tendance',
            line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dash'),
            hoverinfo='skip',
        ))
    except Exception:
        pass

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=x_col.replace('_', ' ').title()),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Prix USD/T'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# 10. Radar chart
@app.callback(
    Output('radar-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('pestel-category', 'value'),
)
def update_radar(json_data, cat):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    if 'produit' not in df.columns:
        return go.Figure()

    # Choisir colonnes normalisables (0-10 scale ou normaliser)
    radar_cols_raw = []
    for category, cols in PESTEL_COLS.items():
        if cat == 'Tous' or cat == category:
            radar_cols_raw += [c for c in cols if c in df.columns]

    if not radar_cols_raw:
        return go.Figure()

    fig = go.Figure()
    for prod in df['produit'].unique():
        sub = df[df['produit'] == prod][radar_cols_raw].mean()
        # Normaliser 0-10
        sub_norm = (sub - df[radar_cols_raw].min()) / (df[radar_cols_raw].max() - df[radar_cols_raw].min() + 1e-9) * 10
        vals = list(sub_norm.values) + [sub_norm.values[0]]
        cols_labels = [c.replace('_', ' ').title() for c in radar_cols_raw] + [radar_cols_raw[0].replace('_', ' ').title()]
        color = PRODUCT_COLORS.get(prod, '#888')
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cols_labels,
            fill='toself',
            name=prod,
            line=dict(color=color, width=2),
            fillcolor=color + '28',
            hovertemplate='<b>' + prod + '</b><br>%{theta}: %{r:.1f}<extra></extra>',
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(size=9, family='JetBrains Mono', color='rgba(255,255,255,0.4)'),
                tickangle=0,
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(size=9, family='JetBrains Mono', color='rgba(255,255,255,0.6)'),
            ),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)'),
        margin=dict(l=30, r=30, t=20, b=20),
    )
    return fig


# 11. Tableau de données
@app.callback(
    Output('data-table-container', 'children'),
    Output('data-info', 'children'),
    Input('filtered-data-store', 'data'),
)
def update_table(json_data):
    df = pd.read_json(json_data, orient='split') if json_data else load_data()
    preview = df.head(50)

    table = dash_table.DataTable(
        data=preview.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in preview.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgba(78,205,196,0.15)',
            'color': '#4ECDC4',
            'fontFamily': 'JetBrains Mono',
            'fontSize': '11px',
            'fontWeight': '600',
            'border': '1px solid rgba(78,205,196,0.2)',
            'letterSpacing': '1px',
            'padding': '10px 12px',
        },
        style_cell={
            'backgroundColor': 'rgba(255,255,255,0.02)',
            'color': 'rgba(255,255,255,0.75)',
            'fontFamily': 'JetBrains Mono',
            'fontSize': '11px',
            'border': '1px solid rgba(255,255,255,0.05)',
            'padding': '8px 12px',
            'maxWidth': '130px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(78,205,196,0.03)'},
        ],
        filter_action='native',
        sort_action='native',
        sort_mode='multi',
    )
    info = f'{len(df):,} lignes × {len(df.columns)} colonnes · Affichage 50 premières'
    return table, info


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')

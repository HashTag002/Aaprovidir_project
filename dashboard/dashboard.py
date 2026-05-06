"""
Generic Data Exploration Dashboard
Module 1 : Dashboard interactif — visualisation données d'entrée
Entièrement générique : fonctionne avec n'importe quel dataset CSV
"""

import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'pestel_agricole.csv'

# Palette de couleurs génériques pour les séries/catégories
SERIES_COLORS = [
    '#4ECDC4', '#E8A838', '#A78BFA', '#34D399',
    '#F87171', '#60A5FA', '#FB923C', '#E879F9',
    '#A3E635', '#FDBA74', '#93C5FD', '#6EE7B7',
]

def get_color(i):
    return SERIES_COLORS[i % len(SERIES_COLORS)]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_default():
    df = pd.read_csv(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def df_to_json(df):
    d = df.copy()
    for c in d.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        d[c] = d[c].astype(str)
    return d.to_json(orient='split')

def json_to_df(j):
    df = pd.read_json(io.StringIO(j), orient='split')
    # Tenter de détecter les colonnes date
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower() or 'année' in c.lower() or 'year' in c.lower():
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().sum() > len(df) * 0.5:
                    df[c] = parsed
            except Exception:
                pass
    return df

def detect_date_col(df):
    """Retourne le nom de la première colonne datetime trouvée."""
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def detect_group_col(df):
    """Retourne la première colonne catégorielle à faible cardinalité (<=30 valeurs)."""
    for c in df.select_dtypes(include=['object', 'category']).columns:
        if 1 < df[c].nunique() <= 30:
            return c
    return None

def get_numeric_cols(df):
    return list(df.select_dtypes(include='number').columns)

def label(col):
    return col.replace('_', ' ').title()

def empty_fig(msg='Aucune donnée'):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.3)'),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=msg, x=0.5, y=0.5, showarrow=False,
                          font=dict(color='rgba(255,255,255,0.2)', size=14))],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig

def base_layout(**extra):
    d = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono', color='rgba(255,255,255,0.7)', size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
    )
    d.update(extra)
    return d

CARD = {
    'background': 'rgba(255,255,255,0.04)',
    'border': '1px solid rgba(255,255,255,0.10)',
    'borderRadius': '12px', 'padding': '20px', 'marginBottom': '16px',
}

def section_title(text, color='#4ECDC4'):
    return html.Div(text, style={
        'fontFamily': 'Syne', 'fontWeight': '700', 'fontSize': '13px',
        'color': color, 'letterSpacing': '1px', 'marginBottom': '12px',
    })

def kpi_card(title, value, unit, color, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '22px', 'marginBottom': '6px'}),
        html.Div(title, style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
            'letterSpacing': '2px', 'color': 'rgba(255,255,255,0.45)',
            'textTransform': 'uppercase', 'marginBottom': '4px',
        }),
        html.Span(value, style={
            'fontFamily': 'Syne', 'fontSize': '22px', 'fontWeight': '800', 'color': color,
        }),
        html.Span(f' {unit}', style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '11px', 'color': 'rgba(255,255,255,0.35)',
        }),
    ], style={
        'background': 'rgba(255,255,255,0.04)',
        'border': '1px solid rgba(255,255,255,0.10)',
        'borderTop': f'3px solid {color}',
        'borderRadius': '12px', 'padding': '16px 20px',
        'textAlign': 'center', 'flex': '1', 'minWidth': '140px',
    })

DROPDOWN_STYLE = {
    'backgroundColor': '#1a2533',
    'color': '#e8eaed',
    'border': '1px solid rgba(255,255,255,0.15)',
    'borderRadius': '6px',
}

# ─── APP ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        'https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
    ],
    suppress_callback_exceptions=True,
    title='Data Explorer Dashboard',
)

# Données par défaut pour initialiser les composants
_df0 = load_default()
_num0 = get_numeric_cols(_df0)
_opts0 = [{'label': label(c), 'value': c} for c in _num0]
_date_col0 = detect_date_col(_df0)
_group_col0 = detect_group_col(_df0)
_dates0 = sorted(_df0[_date_col0].dropna().unique()) if _date_col0 else []
_n0 = len(_dates0)
_marks0 = {i: str(pd.Timestamp(d).year) for i, d in enumerate(_dates0) if i % 12 == 0 or i == _n0 - 1} if _n0 > 0 else {0: '0'}

# ─── LAYOUT ───────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={'background': '#0d1520', 'minHeight': '100vh', 'fontFamily': 'Syne, sans-serif', 'color': '#e8eaed'},
    children=[

    # HEADER
    html.Div(style={
        'background': 'linear-gradient(135deg,#0f1923,#1a2533,#0f1923)',
        'borderBottom': '1px solid rgba(78,205,196,0.3)',
        'padding': '18px 32px', 'display': 'flex',
        'alignItems': 'center', 'justifyContent': 'space-between',
    }, children=[
        html.Div([
            html.Div('DATA EXPLORER', style={
                'fontFamily': 'Syne', 'fontWeight': '800', 'fontSize': '20px',
                'letterSpacing': '3px', 'color': '#4ECDC4',
            }),
            html.Div('Visualisation interactive · Dataset universel', style={
                'fontFamily': 'JetBrains Mono', 'fontSize': '11px',
                'color': 'rgba(255,255,255,0.4)', 'letterSpacing': '1px',
            }),
        ]),
        html.Div(id='dataset-badge', style={
            'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
            'color': 'rgba(255,255,255,0.35)', 'letterSpacing': '2px',
        }),
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

            # Colonne de groupe (catégorielle)
            html.Div([
                html.Div('🏷 Regrouper par', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#A78BFA', 'marginBottom': '6px',
                }),
                dcc.Dropdown(id='group-col-selector',
                    options=[], value=None, clearable=True,
                    placeholder='(aucun)',
                    style={'minWidth': '160px'},
                ),
            ], style={'flex': '1', 'minWidth': '160px'}),

            # Valeur du groupe
            html.Div([
                html.Div('🔍 Filtrer la valeur', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#E8A838', 'marginBottom': '6px',
                }),
                dcc.Dropdown(id='group-value-selector',
                    options=[], value=None, clearable=True,
                    placeholder='Toutes',
                    style={'minWidth': '140px'},
                ),
            ], style={'flex': '1', 'minWidth': '140px'}),

            # Slider date (masqué si pas de col date)
            html.Div(id='date-slider-container', children=[
                html.Div('📅 Période', style={
                    'fontFamily': 'JetBrains Mono', 'fontSize': '10px',
                    'letterSpacing': '2px', 'color': '#34D399', 'marginBottom': '6px',
                }),
                dcc.RangeSlider(
                    id='date-slider', min=0, max=max(_n0 - 1, 0), step=1,
                    value=[0, max(_n0 - 1, 0)], marks=_marks0,
                    tooltip={'placement': 'bottom', 'always_visible': False},
                ),
            ], style={'flex': '3', 'minWidth': '280px'}),
        ]),
    ]),

    # KPI
    html.Div(id='kpi-section', style={'padding': '0 32px'}),

    # GRAPHIQUES
    html.Div(style={'padding': '0 32px 32px'}, children=[

        # Ligne 1 : Série temporelle (ou index)
        dbc.Row([dbc.Col([html.Div(style=CARD, children=[
            html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap',
                            'alignItems': 'center', 'marginBottom': '12px'}, children=[
                section_title('📈 Séries temporelles', '#4ECDC4'),
                html.Div([
                    dcc.Dropdown(id='ts-y-cols',
                        options=_opts0,
                        value=_num0[:3] if len(_num0) >= 3 else _num0,
                        multi=True,
                        placeholder='Choisir des colonnes…',
                        style={'minWidth': '320px'},
                    ),
                ]),
            ]),
            dcc.Graph(id='timeseries-chart', config={'displayModeBar': True}, style={'height': '320px'}),
        ])], md=12)]),

        # Ligne 2 : Scatter + Distribution
        dbc.Row([
            dbc.Col([html.Div(style=CARD, children=[
                section_title('🔭 Nuage de points', '#4ECDC4'),
                html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '10px'}, children=[
                    dcc.Dropdown(id='scatter-x', options=_opts0,
                        value=_num0[0] if _num0 else None, clearable=False,
                        placeholder='Axe X', style={'flex': '1', 'minWidth': '140px'}),
                    dcc.Dropdown(id='scatter-y', options=_opts0,
                        value=_num0[1] if len(_num0) > 1 else (_num0[0] if _num0 else None),
                        clearable=False, placeholder='Axe Y',
                        style={'flex': '1', 'minWidth': '140px'}),
                ]),
                dcc.Graph(id='scatter-chart', config={'displayModeBar': False}, style={'height': '300px'}),
            ])], md=7),

            dbc.Col([html.Div(style=CARD, children=[
                section_title('📊 Distribution', '#F87171'),
                dcc.Dropdown(id='dist-col', options=_opts0,
                    value=_num0[0] if _num0 else None, clearable=False,
                    style={'marginBottom': '10px'}),
                dcc.Graph(id='distribution-chart', config={'displayModeBar': False}, style={'height': '300px'}),
            ])], md=5),
        ]),

        # Ligne 3 : Heatmap corrélation + Donut/Bar catégoriel
        dbc.Row([
            dbc.Col([html.Div(style=CARD, children=[
                section_title('🔥 Matrice de corrélation', '#34D399'),
                html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '10px'}, children=[
                    dcc.Dropdown(id='heatmap-cols',
                        options=_opts0,
                        value=_num0[:8] if len(_num0) >= 8 else _num0,
                        multi=True,
                        placeholder='Colonnes à corréler…',
                        style={'flex': '1', 'minWidth': '260px'}),
                ]),
                dcc.Graph(id='correlation-heatmap', config={'displayModeBar': False}, style={'height': '380px'}),
            ])], md=7),

            dbc.Col([html.Div(style=CARD, children=[
                section_title('🥧 Agrégation par groupe', '#A78BFA'),
                html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '10px'}, children=[
                    dcc.Dropdown(id='agg-col',
                        options=_opts0,
                        value=_num0[0] if _num0 else None,
                        clearable=False,
                        placeholder='Colonne à agréger',
                        style={'flex': '1', 'minWidth': '140px'}),
                    dcc.Dropdown(id='agg-func',
                        options=[
                            {'label': 'Moyenne', 'value': 'mean'},
                            {'label': 'Somme', 'value': 'sum'},
                            {'label': 'Médiane', 'value': 'median'},
                            {'label': 'Max', 'value': 'max'},
                            {'label': 'Min', 'value': 'min'},
                        ],
                        value='mean', clearable=False,
                        style={'flex': '1', 'minWidth': '120px'}),
                ]),
                dcc.Graph(id='agg-chart', config={'displayModeBar': False}, style={'height': '340px'}),
            ])], md=5),
        ]),

        # Ligne 4 : Box plot multi-colonnes
        dbc.Row([dbc.Col([html.Div(style=CARD, children=[
            section_title('📦 Box plots comparatifs', '#E8A838'),
            html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '10px'}, children=[
                dcc.Dropdown(id='box-cols',
                    options=_opts0,
                    value=_num0[:5] if len(_num0) >= 5 else _num0,
                    multi=True,
                    placeholder='Colonnes à afficher…',
                    style={'flex': '1', 'minWidth': '260px'}),
                dcc.Dropdown(id='box-type',
                    options=[
                        {'label': 'Box plot', 'value': 'box'},
                        {'label': 'Violin', 'value': 'violin'},
                    ],
                    value='box', clearable=False,
                    style={'minWidth': '130px'}),
            ]),
            dcc.Graph(id='box-chart', config={'displayModeBar': True}, style={'height': '320px'}),
        ])], md=12)]),

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

    # Stores
    dcc.Store(id='filtered-data-store', data=df_to_json(_df0)),
    dcc.Store(id='raw-data-store', data=None),
    dcc.Store(id='meta-store', data={
        'date_col': _date_col0,
        'group_col': _group_col0,
        'num_cols': _num0,
        'dates': [str(d) for d in _dates0],
    }),
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
        # Détecter colonnes date
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                try:
                    df[c] = pd.to_datetime(df[c], errors='coerce')
                except Exception:
                    pass
        return df_to_json(df), f'✓ {filename} — {len(df):,} lignes · {len(df.columns)} colonnes'
    except Exception as e:
        return dash.no_update, f'⚠ Erreur : {e}'


# 2. Nouveau dataset → mise à jour meta + contrôles
@app.callback(
    Output('meta-store', 'data'),
    Output('group-col-selector', 'options'),
    Output('group-col-selector', 'value'),
    Output('date-slider', 'max'),
    Output('date-slider', 'marks'),
    Output('date-slider', 'value'),
    Output('dataset-badge', 'children'),
    Input('raw-data-store', 'data'),
    prevent_initial_call=True,
)
def update_meta(raw_json):
    if not raw_json:
        return dash.no_update, [], None, 0, {0: '0'}, [0, 0], ''
    df = json_to_df(raw_json)
    date_col = detect_date_col(df)
    group_col = detect_group_col(df)
    num_cols = get_numeric_cols(df)
    dates = sorted(df[date_col].dropna().unique()) if date_col else []
    n = len(dates)
    marks = {}
    if n > 0:
        for i, d in enumerate(dates):
            if i % max(1, n // 8) == 0 or i == n - 1:
                marks[i] = str(pd.Timestamp(d).year if hasattr(pd.Timestamp(d), 'year') else d)
    else:
        marks = {0: '0'}

    # Options de groupe
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if 1 < df[c].nunique() <= 30]
    group_opts = [{'label': label(c), 'value': c} for c in cat_cols]

    meta = {
        'date_col': date_col,
        'group_col': group_col,
        'num_cols': num_cols,
        'dates': [str(d) for d in dates],
    }
    badge = f'● {len(df):,} lignes · {len(df.columns)} col · {len(num_cols)} numériques'
    return meta, group_opts, group_col, max(n - 1, 0), marks, [0, max(n - 1, 0)], badge


# 3. Options valeur du groupe — lit UNIQUEMENT raw-data-store pour éviter le cycle
@app.callback(
    Output('group-value-selector', 'options'),
    Output('group-value-selector', 'value'),
    Input('group-col-selector', 'value'),
    Input('raw-data-store', 'data'),
)
def update_group_values(group_col, raw_json):
    if not group_col:
        return [], None
    try:
        df = json_to_df(raw_json) if raw_json else load_default()
        vals = sorted(df[group_col].dropna().unique().tolist())
        opts = [{'label': str(v), 'value': str(v)} for v in vals]
        return opts, None
    except Exception:
        return [], None


# 4. Filtres → filtered-data-store + màj des dropdowns de colonnes
@app.callback(
    Output('filtered-data-store', 'data'),
    Output('ts-y-cols', 'options'),
    Output('ts-y-cols', 'value'),
    Output('scatter-x', 'options'),
    Output('scatter-x', 'value'),
    Output('scatter-y', 'options'),
    Output('scatter-y', 'value'),
    Output('dist-col', 'options'),
    Output('dist-col', 'value'),
    Output('heatmap-cols', 'options'),
    Output('heatmap-cols', 'value'),
    Output('agg-col', 'options'),
    Output('agg-col', 'value'),
    Output('box-cols', 'options'),
    Output('box-cols', 'value'),
    Input('raw-data-store', 'data'),
    Input('group-col-selector', 'value'),
    Input('group-value-selector', 'value'),
    Input('date-slider', 'value'),
    Input('meta-store', 'data'),
)
def filter_data(raw_json, group_col, group_val, date_range, meta):
    df = json_to_df(raw_json) if raw_json else load_default()
    date_col = meta.get('date_col') if meta else detect_date_col(df)
    dates = meta.get('dates', []) if meta else []

    # Filtrer par groupe
    if group_col and group_val and group_col in df.columns:
        df = df[df[group_col].astype(str) == str(group_val)]

    # Filtrer par date
    if date_col and dates and date_range and len(dates) > 0:
        s_idx = max(0, min(date_range[0], len(dates) - 1))
        e_idx = max(0, min(date_range[1], len(dates) - 1))
        try:
            d_start = pd.Timestamp(dates[s_idx])
            d_end = pd.Timestamp(dates[e_idx])
            df = df[(df[date_col] >= d_start) & (df[date_col] <= d_end)]
        except Exception:
            pass

    num_cols = get_numeric_cols(df)
    opts = [{'label': label(c), 'value': c} for c in num_cols]

    def safe_val(n, idx=0):
        return num_cols[idx] if len(num_cols) > idx else (num_cols[0] if num_cols else None)

    ts_val = num_cols[:3] if len(num_cols) >= 3 else num_cols
    hm_val = num_cols[:8] if len(num_cols) >= 8 else num_cols
    box_val = num_cols[:5] if len(num_cols) >= 5 else num_cols

    return (
        df_to_json(df),
        opts, ts_val,
        opts, safe_val(num_cols, 0),
        opts, safe_val(num_cols, 1),
        opts, safe_val(num_cols, 0),
        opts, hm_val,
        opts, safe_val(num_cols, 0),
        opts, box_val,
    )


# 5. KPI generiques
@app.callback(Output('kpi-section', 'children'), Input('filtered-data-store', 'data'), Input('meta-store', 'data'))
def update_kpis(json_data, meta):
    if not json_data:
        return []
    try:
        df = json_to_df(json_data)
        cards = []
        num_cols = get_numeric_cols(df)
        # Lignes
        cards.append(kpi_card('Lignes', f'{len(df):,}', '', '#4ECDC4', '🗂'))
        # Colonnes
        cards.append(kpi_card('Colonnes', str(len(df.columns)), '', '#A78BFA', '📐'))
        # Premières colonnes numériques avec stat rapide
        for i, c in enumerate(num_cols[:4]):
            color = get_color(i + 2)
            mean_val = df[c].mean()
            if abs(mean_val) >= 1000:
                fmt = f'{mean_val:,.0f}'
            elif abs(mean_val) >= 1:
                fmt = f'{mean_val:.2f}'
            else:
                fmt = f'{mean_val:.4f}'
            cards.append(kpi_card(f'Moy. {label(c)[:14]}', fmt, '', color, '📊'))

        # Périodes si date dispo
        date_col = meta.get('date_col') if meta else None
        if date_col and date_col in df.columns:
            cards.append(kpi_card('Périodes', str(df[date_col].nunique()), '', '#34D399', '📅'))

        return html.Div(children=cards,
                        style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '16px'})
    except Exception:
        return []


# 6. Série temporelle (ou index si pas de date)
@app.callback(
    Output('timeseries-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('ts-y-cols', 'value'),
    Input('meta-store', 'data'),
    Input('group-col-selector', 'value'),
)
def update_timeseries(json_data, y_cols, meta, group_col):
    if not json_data or not y_cols:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        cols = [y_cols] if isinstance(y_cols, str) else list(y_cols)
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return empty_fig()

        date_col = meta.get('date_col') if meta else detect_date_col(df)
        x_axis = df[date_col] if date_col and date_col in df.columns else df.index

        fig = go.Figure()

        if group_col and group_col in df.columns and len(cols) == 1:
            # Mode : une colonne, séries par groupe
            col = cols[0]
            for i, grp in enumerate(sorted(df[group_col].dropna().unique())):
                sub = df[df[group_col] == grp].sort_values(date_col) if date_col else df[df[group_col] == grp]
                x = sub[date_col] if date_col and date_col in sub.columns else sub.index
                fig.add_trace(go.Scatter(
                    x=x, y=sub[col], name=str(grp),
                    mode='lines', line=dict(color=get_color(i), width=2),
                ))
        else:
            # Mode : plusieurs colonnes
            for i, col in enumerate(cols):
                agg = df.groupby(date_col)[col].mean().reset_index().sort_values(date_col) if date_col and date_col in df.columns else df[[col]].reset_index()
                x = agg[date_col] if date_col and date_col in agg.columns else agg.index
                fig.add_trace(go.Scatter(
                    x=x, y=agg[col], name=label(col),
                    mode='lines', line=dict(color=get_color(i), width=2),
                ))

        layout = base_layout()
        layout.update(dict(
            hovermode='x unified',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            legend=dict(bgcolor='rgba(13,21,32,0.9)', orientation='h',
                        yanchor='bottom', y=1.02, xanchor='left', x=0,
                        bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
            margin=dict(l=10, r=10, t=40, b=10),
        ))
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig(f'Erreur : {e}')


# 7. Scatter
@app.callback(
    Output('scatter-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('scatter-x', 'value'),
    Input('scatter-y', 'value'),
    Input('group-col-selector', 'value'),
)
def update_scatter(json_data, x_col, y_col, group_col):
    if not json_data or not x_col or not y_col:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if x_col not in df.columns or y_col not in df.columns:
            return empty_fig()
        fig = go.Figure()

        if group_col and group_col in df.columns:
            groups = sorted(df[group_col].dropna().unique())
            for i, grp in enumerate(groups):
                sub = df[df[group_col] == grp]
                fig.add_trace(go.Scatter(
                    x=sub[x_col], y=sub[y_col], mode='markers', name=str(grp),
                    marker=dict(color=get_color(i), size=5, opacity=0.75,
                                line=dict(width=0.5, color='rgba(255,255,255,0.1)')),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[y_col], mode='markers', name='Points',
                marker=dict(color='#4ECDC4', size=5, opacity=0.7,
                            line=dict(width=0.5, color='rgba(255,255,255,0.1)')),
            ))
        # Trend
        try:
            x_v = df[x_col].dropna()
            y_v = df.loc[x_v.index, y_col].dropna()
            idx = x_v.index.intersection(y_v.index)
            coeffs = np.polyfit(x_v[idx], y_v[idx], 1)
            xr = np.linspace(x_v.min(), x_v.max(), 100)
            fig.add_trace(go.Scatter(
                x=xr, y=np.polyval(coeffs, xr), mode='lines', name='Tendance',
                line=dict(color='rgba(255,255,255,0.4)', width=1.5, dash='dash'),
                hoverinfo='skip',
            ))
        except Exception:
            pass
        layout = base_layout()
        layout.update(
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=label(x_col)),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=label(y_col)),
        )
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig(f'Erreur : {e}')


# 8. Distribution
@app.callback(
    Output('distribution-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('dist-col', 'value'),
    Input('group-col-selector', 'value'),
)
def update_distribution(json_data, col, group_col):
    if not json_data or not col:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if col not in df.columns:
            return empty_fig()
        fig = go.Figure()

        if group_col and group_col in df.columns:
            groups = sorted(df[group_col].dropna().unique())
            for i, grp in enumerate(groups):
                vals = df[df[group_col] == grp][col].dropna()
                fig.add_trace(go.Histogram(
                    x=vals, name=str(grp), nbinsx=20,
                    marker_color=get_color(i), opacity=0.65,
                ))
            layout = base_layout()
            layout.update(barmode='overlay', bargap=0.05,
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=label(col)),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Fréquence'),
            )
        else:
            vals = df[col].dropna()
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=25, marker_color='#4ECDC4', opacity=0.8,
            ))
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
            layout.update(showlegend=False, bargap=0.05,
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title=label(col)),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Fréquence'),
            )
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig(f'Erreur : {e}')


# 9. Heatmap corrélation
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('heatmap-cols', 'value'),
)
def update_heatmap(json_data, cols):
    if not json_data or not cols:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        cols = [c for c in (cols if isinstance(cols, list) else [cols]) if c in df.columns]
        if len(cols) < 2:
            return empty_fig('Sélectionnez au moins 2 colonnes')
        corr = df[cols].corr()
        labels = [c.replace('_', '<br>') for c in corr.columns]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            colorscale=[[0, '#E8A838'], [0.5, '#1a2533'], [1, '#4ECDC4']],
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text:.2f}',
            textfont=dict(size=9, family='JetBrains Mono'),
            hovertemplate='%{y} × %{x} : %{z:.3f}<extra></extra>',
        ))
        layout = base_layout()
        layout.update(xaxis=dict(side='bottom', tickangle=-45, tickfont=dict(size=9)))
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig(f'Erreur : {e}')


# 10. Agrégation par groupe (bar / donut)
@app.callback(
    Output('agg-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('agg-col', 'value'),
    Input('agg-func', 'value'),
    Input('group-col-selector', 'value'),
    Input('meta-store', 'data'),
)
def update_agg(json_data, agg_col, agg_func, group_col, meta):
    if not json_data or not agg_col:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        if agg_col not in df.columns:
            return empty_fig()

        # Choisir colonne de groupe
        grp = group_col if group_col and group_col in df.columns else detect_group_col(df)
        if not grp:
            # Pas de colonne cat : histogramme simple
            return empty_fig('Sélectionnez une colonne de regroupement')

        agg = df.groupby(grp)[agg_col].agg(agg_func).reset_index().sort_values(agg_col, ascending=False)
        colors = [get_color(i) for i in range(len(agg))]

        if len(agg) <= 8:
            fig = go.Figure(go.Pie(
                labels=agg[grp], values=agg[agg_col], hole=0.5,
                marker_colors=colors,
                textinfo='label+percent',
                textfont=dict(family='JetBrains Mono', size=10),
                hovertemplate='<b>%{label}</b> : %{value:.2f}<extra></extra>',
            ))
            layout = base_layout()
            layout.update(showlegend=False)
        else:
            fig = go.Figure(go.Bar(
                x=agg[grp], y=agg[agg_col],
                marker_color=colors,
                hovertemplate='<b>%{x}</b> : %{y:.2f}<extra></extra>',
            ))
            layout = base_layout()
            layout.update(
                xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                showlegend=False,
            )
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig(f'Erreur : {e}')


# 11. Box plots
@app.callback(
    Output('box-chart', 'figure'),
    Input('filtered-data-store', 'data'),
    Input('box-cols', 'value'),
    Input('box-type', 'value'),
    Input('group-col-selector', 'value'),
)
def update_box(json_data, cols, box_type, group_col):
    if not json_data or not cols:
        return empty_fig()
    try:
        df = json_to_df(json_data)
        cols = [c for c in (cols if isinstance(cols, list) else [cols]) if c in df.columns]
        if not cols:
            return empty_fig()
        fig = go.Figure()

        if group_col and group_col in df.columns and len(cols) == 1:
            col = cols[0]
            groups = sorted(df[group_col].dropna().unique())
            for i, grp in enumerate(groups):
                vals = df[df[group_col] == grp][col].dropna()
                if box_type == 'violin':
                    fig.add_trace(go.Violin(y=vals, name=str(grp), line_color=get_color(i),
                                            fillcolor=get_color(i) + '30', box_visible=True, meanline_visible=True))
                else:
                    fig.add_trace(go.Box(y=vals, name=str(grp), marker_color=get_color(i)))
        else:
            # Normalisation z-score si plusieurs colonnes très différentes
            for i, col in enumerate(cols):
                vals = df[col].dropna()
                if box_type == 'violin':
                    fig.add_trace(go.Violin(y=vals, name=label(col), line_color=get_color(i),
                                            fillcolor=get_color(i) + '30', box_visible=True, meanline_visible=True))
                else:
                    fig.add_trace(go.Box(y=vals, name=label(col), marker_color=get_color(i)))

        layout = base_layout()
        layout.update(
            showlegend=True,
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
            xaxis=dict(tickangle=-20),
        )
        fig.update_layout(**layout)
        return fig
    except Exception as e:
        return empty_fig(f'Erreur : {e}')


# 12. Tableau
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
        # Formater les dates pour affichage
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime('%Y-%m-%d').fillna('')
        preview = df.head(100)
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
                'padding': '8px 12px', 'maxWidth': '150px',
                'overflow': 'hidden', 'textOverflow': 'ellipsis',
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(78,205,196,0.025)'},
            ],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
        )
        info = f'{len(df):,} lignes × {len(df.columns)} colonnes · 100 premières affichées'
        return table, info
    except Exception as e:
        return html.Div(f'Erreur : {e}', style={'color': '#F87171', 'fontFamily': 'JetBrains Mono'}), ''


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050, host='0.0.0.0')

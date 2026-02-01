#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:48:50 2025

@author: henrik
"""



#JAG TRÄNAR MIN DATA PÅ PGA CHAMPIONSHIP RESULTAT!!!!!

#Detta är en statistisk modell med viktade samband beroende på rankingen i kategorierna
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df_merge = pd.read_csv("df_merge.csv")

# Features vi vill använda
features = ['match_score', 'SG: TOT', 'FedEx_Points']

# Skala features till 0-1
scaler = MinMaxScaler()
df_merge_scaled = df_merge.copy()  # gör kopia för att undvika SettingWithCopyWarning
df_merge_scaled[features] = scaler.fit_transform(df_merge[features])

# Vikter för varje feature (justera enligt kolumnnamn)
weights = {
    'match_score': 0.5,
    'SG: TOT': 0.2,
    'FedEx_Points': 0.3,
}

# Beräkna totalpoäng som viktad summa
df_merge_scaled['total_score'] = (
    df_merge_scaled['match_score'] * weights['match_score'] +
    df_merge_scaled['SG: TOT'] * weights['SG: TOT'] +
    df_merge_scaled['FedEx_Points'] * weights['FedEx_Points']
)

# Sortera spelare på totalpoängen, högst först
df_merge_scaled = df_merge_scaled.sort_values(by='total_score', ascending=False).reset_index(drop=True)

print(df_merge_scaled[['Spelare', 'total_score']].head(20))





#%%
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_and_process_data():
    # Simulerad datainläsning - ersätt med din faktiska implementation
    data = {
        'Spelare': [f'Spelare {i}' for i in range(1, 31)],
        'SG: OTT': np.random.uniform(-2, 5, 30),
        'SG: APP': np.random.uniform(-2, 5, 30),
        'SG: ATG': np.random.uniform(-2, 5, 30),
        'SG: P': np.random.uniform(-2, 5, 30),
        'SG: TOT': np.random.uniform(-2, 10, 30),
        'FedEx_Points': np.random.randint(300, 3500, 30),
        'Position': np.random.randint(1, 100, 30),
        'Top20': np.random.randint(0, 2, 30)
    }
    return pd.DataFrame(data)

def calculate_match_scores(df, weights, score_col_name):
    cols = list(weights.keys())
    df_clean = df.dropna(subset=cols).copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean[score_col_name] = sum(df_clean[col] * weight for col, weight in weights.items())
    return df_clean

def calculate_probabilities(probability_series):
    raw_prob = np.clip(probability_series, 0.01, 0.99)
    scaled_prob = 0.05 + 0.90 * (raw_prob - np.min(raw_prob)) / (np.max(raw_prob) - np.min(raw_prob) + 1e-10)
    return np.clip(scaled_prob, 0.05, 0.95)

def train_model(data):
    # Beräkna viktade poäng
    weights_oakmont = {'SG: OTT': 0.8, 'SG: APP': 0.6, 'SG: ATG': 0.3, 'SG: P': 0.5}
    weights_pga = {'SG: OTT': 0.4, 'SG: APP': 0.8, 'SG: ATG': 0.7, 'SG: P': 0.8}
    
    data = calculate_match_scores(data, weights_oakmont, 'match_score_oakmont')
    data = calculate_match_scores(data, weights_pga, 'match_score_pga')
    data['super_score'] = (data['match_score_oakmont'] + data['match_score_pga']) / 2
    
    # Rensa data
    cleaned_data = data[
        ~((data['super_score'] > 7) & (data['Position'] > 40)) & 
        ~((data['super_score'] < 3) & (data['Position'] <= 20))
    ].copy()
    
    # Hantera fall där vi bara har en klass
    if cleaned_data['Top20'].nunique() < 2:
        print("Varning: Endast en klass i data, använder fallback-metod")
        cleaned_data['Top20_prob'] = np.where(
            cleaned_data['super_score'] > cleaned_data['super_score'].median(),
            0.7, 0.3
        )
    else:
        # Träna modell om vi har båda klasserna
        X = cleaned_data[['super_score']]
        y = cleaned_data['Top20']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        model = LogisticRegression(
            class_weight='balanced',
            penalty='l2',
            C=0.1,
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        cleaned_data['Top20_prob'] = model.predict_proba(X)[:, 1]
    
    cleaned_data['Top20_prob'] = calculate_probabilities(cleaned_data['Top20_prob'])
    return cleaned_data.sort_values('Top20_prob', ascending=False)

def create_visualizations(df):
    # 1. Jämförelse av båda match_scores
    score_comparison_fig = px.scatter(
        df,
        x='match_score_oakmont',
        y='match_score_pga',
        color='Top20',
        hover_name='Spelare',
        title='Match Scores: Oakmont vs PGA'
    )
    
    # 2. Super Score fördelning
    hist_fig = px.histogram(
        df, 
        x='super_score', 
        color='Top20',
        nbins=20,
        title='Fördelning av Super Score'
    )
    
    # 3. Top20 sannolikhet
    scatter_fig = px.scatter(
        df,
        x='super_score',
        y='Top20_prob',
        color='Position',
        size='FedEx_Points',
        hover_name='Spelare',
        title='Super Score vs Top20 Sannolikhet'
    )
    
    return score_comparison_fig, hist_fig, scatter_fig

# Initiera app
app = dash.Dash(__name__)

# Ladda och processa data
raw_data = load_and_process_data()
result_data = train_model(raw_data)

# Skapa visualiseringar
score_comparison_fig, hist_fig, scatter_fig = create_visualizations(result_data)

# App layout
app.layout = html.Div([
    html.H1("Golf Prestanda Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Graph(figure=score_comparison_fig)
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(figure=hist_fig)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=scatter_fig)
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    html.H2("Topp 20 Spelare", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='top-players-table',
        columns=[
            {'name': 'Spelare', 'id': 'Spelare'},
            {'name': 'Oakmont Score', 'id': 'match_score_oakmont', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'PGA Score', 'id': 'match_score_pga', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Top20 Sannolikhet', 'id': 'Top20_prob', 'type': 'numeric', 'format': {'specifier': '.1%'}}
        ],
        data=result_data.head(20).to_dict('records'),
        style_table={'width': '90%', 'margin': '0 auto'},
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white'}
    )
])

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
#%%


import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =============================================
# HÄMTA OCH BEARBETA DATA (FRÅN DIN ORIGINALKOD)
# =============================================

def load_and_process_data():
    # Läs in data från HTML-filer
    df_list = pd.read_html("golfprojekt_strokes_gained.html")
    df_fedex_list = pd.read_html("fedex_top100.html")
    pga_df_list = pd.read_html("pga_championship.html")
    
    df = df_list[0]
    df_fedex = df_fedex_list[0]
    pga_df = pga_df_list[0]
    
    # Bearbeta data
    common_players = set(df['Spelare']) & set(df_fedex['Spelare']) & set(pga_df['Spelare'])
    
    df_sg = df[["Spelare", "SG: OTT", "SG: APP", "SG: ATG", "SG: P", "SG: TOT"]].copy()
    df_sg = df_sg[df_sg['Spelare'].isin(common_players)]
    
    for col in df_sg.columns[1:]:
        df_sg.loc[:, col] = pd.to_numeric(df_sg[col], errors="coerce")
    
    df_sg = df_sg.dropna().sort_values(by='SG: TOT', ascending=False)
    
    # Beräkna matchpoäng
    weights = {'SG: OTT': 0.8, 'SG: APP': 0.6, 'SG: ATG': 0.3, 'SG: P': 0.5}
    df_with_scores = calculate_match_scores(df, weights)
    df_with_scores = df_with_scores[df_with_scores['Spelare'].isin(common_players)]
    
    # Slå ihop data
    final_df = pd.merge(
        df_with_scores[['Spelare', 'match_score']],
        df_sg[['Spelare', 'SG: TOT']],
        on='Spelare',
        how='inner'
    )
    
    final_df = pd.merge(
        final_df,
        df_fedex[df_fedex['Spelare'].isin(common_players)][['Spelare', 'FedEx_Points']],
        on='Spelare',
        how='inner'
    )
    
    final_df = pd.merge(
        final_df,
        pga_df[['Spelare', 'Position', 'Top20']],
        on='Spelare',
        how='inner'
    )
    
    return final_df

def calculate_match_scores(df, weights):
    cols = list(weights.keys())
    df_clean = df.dropna(subset=cols).copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['match_score'] = sum(df_clean[col] * weight for col, weight in weights.items())
    return df_clean

def train_model(data):
    # Beräkna super_score
    data['super_score'] = data.apply(calculate_player_score, axis=1)
    min_score, max_score = data['super_score'].min(), data['super_score'].max()
    data['super_score'] = 1 + ((data['super_score'] - min_score) / (max_score - min_score)) * 9
    
    # Rensa data
    cleaned_data = data[
        ~((data['super_score'] > 7) & (data['Position'] > 40)) & 
        ~((data['super_score'] < 3) & (data['Position'] <= 20))
    ].copy()
    
    # Träna modell
    X = cleaned_data[['super_score']]
    y = cleaned_data['Top20']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        C=0.1,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    cleaned_data['Top20_prob'] = model.predict_proba(X)[:, 1]
    cleaned_data['Top20_prob'] = calculate_probabilities(cleaned_data['Top20_prob'])
    
    return cleaned_data.sort_values('Top20_prob', ascending=False)

def calculate_player_score(player):
    fedex_normalized = (player['FedEx_Points'] - 250) / (3800 - 250)
    return 0.4 * player['match_score'] + 0.4 * player['SG: TOT'] + 0.2 * fedex_normalized

def calculate_probabilities(probability_series):
    raw_prob = np.clip(probability_series, 0.01, 0.99)
    scaled_prob = 0.05 + 0.90 * (raw_prob - np.min(raw_prob)) / (np.max(raw_prob) - np.min(raw_prob) + 1e-10)
    return np.clip(scaled_prob, 0.05, 0.95)

# =============================================
# DASH APP
# =============================================

app = dash.Dash(__name__)

# Ladda och processa data
raw_data = load_and_process_data()
result_data = train_model(raw_data)

# Skapa visualiseringar
def create_visualizations(df):
    # Histogram över super_score
    hist_fig = px.histogram(
        df, 
        x='super_score', 
        color='Top20',
        nbins=20,
        title='Fördelning av Super Score',
        labels={'super_score': 'Super Score', 'count': 'Antal spelare'}
    )
    
    # Spridningsdiagram
    scatter_fig = px.scatter(
        df,
        x='super_score',
        y='Top20_prob',
        color='Position',
        size='FedEx_Points',
        hover_name='Spelare',
        title='Super Score vs Top20 Sannolikhet',
        labels={'super_score': 'Super Score', 'Top20_prob': 'Top20 Sannolikhet'}
    )
    
    # Eftersom vi bara har SG: TOT, skapa ett enkelt stapeldiagram för denna
    if 'SG: TOT' in df.columns:
        sg_fig = px.bar(
            df.sort_values('SG: TOT', ascending=False).head(20),
            x='Spelare',
            y='SG: TOT',
            title='Total Strokes Gained (SG: TOT) för Topp 20 Spelare',
            labels={'SG: TOT': 'Total Strokes Gained'}
        )
    else:
        sg_fig = px.bar(title='Ingen Strokes Gained-data tillgänglig')
    
    return hist_fig, scatter_fig, sg_fig

hist_fig, scatter_fig, sg_fig = create_visualizations(result_data)

# App layout
app.layout = html.Div([
    html.H1("Golf Prestanda Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            dcc.Graph(figure=hist_fig)
        ], style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=scatter_fig)
        ], style={'width': '49%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(figure=sg_fig)
    ], style={'marginTop': '20px'}),
    
    html.H2("Topp 20 Prediktioner för US Open", style={'textAlign': 'center', 'marginTop': '40px'}),
    
    dash_table.DataTable(
        id='top-players-table',
        columns=[
            {'name': 'Spelare', 'id': 'Spelare'},
            {'name': 'Super Score', 'id': 'super_score', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Top20 Sannolikhet', 'id': 'Top20_prob', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            #{'name': 'Position', 'id': 'Position'},
            {'name': 'FedEx Poäng', 'id': 'FedEx_Points'}
        ],
        data=result_data.head(20).to_dict('records'),
        style_table={'overflowX': 'auto', 'width': '90%', 'margin': '0 auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'border': '1px solid #ddd'
        },
        style_header={
            'backgroundColor': '#2c3e50',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {
                    'filter_query': '{Top20_prob} >= 0.7',
                    'column_id': 'Top20_prob'
                },
                'backgroundColor': '#d4edda',
                'color': 'black'
            }
        ]
    )
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

if __name__ == '__main__':
    app.run(debug=True, port=8050)


#%%


import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =============================================
# HÄMTA OCH BEARBETA DATA (FRÅN DIN ORIGINALKOD)
# =============================================

def load_and_process_data():
    # Läs in data från HTML-filer
    df_list = pd.read_html("golfprojekt_strokes_gained.html")
    df_fedex_list = pd.read_html("fedex_top100.html")
    pga_df_list = pd.read_html("pga_championship.html")
    
    df = df_list[0]
    df_fedex = df_fedex_list[0]
    pga_df = pga_df_list[0]
    
    # Bearbeta data
    common_players = set(df['Spelare']) & set(df_fedex['Spelare']) & set(pga_df['Spelare'])
    
    df_sg = df[["Spelare", "SG: OTT", "SG: APP", "SG: ATG", "SG: P", "SG: TOT"]].copy()
    df_sg = df_sg[df_sg['Spelare'].isin(common_players)]
    
    for col in df_sg.columns[1:]:
        df_sg.loc[:, col] = pd.to_numeric(df_sg[col], errors="coerce")
    
    df_sg = df_sg.dropna().sort_values(by='SG: TOT', ascending=False)
    
    # Beräkna matchpoäng
    weights = {'SG: OTT': 0.8, 'SG: APP': 0.6, 'SG: ATG': 0.3, 'SG: P': 0.5}
    df_with_scores = calculate_match_scores(df, weights)
    df_with_scores = df_with_scores[df_with_scores['Spelare'].isin(common_players)]
    
    # Slå ihop data
    final_df = pd.merge(
        df_with_scores[['Spelare', 'match_score']],
        df_sg[['Spelare', 'SG: TOT']],
        on='Spelare',
        how='inner'
    )
    
    final_df = pd.merge(
        final_df,
        df_fedex[df_fedex['Spelare'].isin(common_players)][['Spelare', 'FedEx_Points']],
        on='Spelare',
        how='inner'
    )
    
    final_df = pd.merge(
        final_df,
        pga_df[['Spelare', 'Position', 'Top20']],
        on='Spelare',
        how='inner'
    )
    
    return final_df

def calculate_match_scores(df, weights):
    cols = list(weights.keys())
    df_clean = df.dropna(subset=cols).copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['match_score'] = sum(df_clean[col] * weight for col, weight in weights.items())
    return df_clean

def train_model(data):
    # Beräkna super_score
    data['super_score'] = data.apply(calculate_player_score, axis=1)
    min_score, max_score = data['super_score'].min(), data['super_score'].max()
    data['super_score'] = 1 + ((data['super_score'] - min_score) / (max_score - min_score)) * 9
    
    # Rensa data
    cleaned_data = data[
        ~((data['super_score'] > 7) & (data['Position'] > 40)) & 
        ~((data['super_score'] < 3) & (data['Position'] <= 20))
    ].copy()
    
    # Träna modell
    X = cleaned_data[['super_score']]
    y = cleaned_data['Top20']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        C=0.1,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    cleaned_data['Top20_prob'] = model.predict_proba(X)[:, 1]
    cleaned_data['Top20_prob'] = calculate_probabilities(cleaned_data['Top20_prob'])
    
    return cleaned_data.sort_values('Top20_prob', ascending=False)

def calculate_player_score(player):
    fedex_normalized = (player['FedEx_Points'] - 250) / (3800 - 250)
    return 0.4 * player['match_score'] + 0.4 * player['SG: TOT'] + 0.2 * fedex_normalized

def calculate_probabilities(probability_series):
    raw_prob = np.clip(probability_series, 0.01, 0.99)
    scaled_prob = 0.05 + 0.90 * (raw_prob - np.min(raw_prob)) / (np.max(raw_prob) - np.min(raw_prob) + 1e-10)
    return np.clip(scaled_prob, 0.05, 0.95)

# =============================================
# DASH APP
# =============================================

app = dash.Dash(__name__)

# Ladda och processa data
raw_data = load_and_process_data()
result_data = train_model(raw_data)

# Skapa visualiseringar
def create_visualizations(df):
    # 1. Histogram över super_score
    hist_fig = px.histogram(
        df, 
        x='super_score', 
        color='Top20',
        nbins=20,
        title='Fördelning av Super Score',
        labels={'super_score': 'Super Score', 'count': 'Antal spelare'}
    )
    
    # 2. Spridningsdiagram för Top20 prediktioner
    scatter_fig = px.scatter(
        df,
        x='super_score',
        y='Top20_prob',
        color='Position',
        size='FedEx_Points',
        hover_name='Spelare',
        title='Super Score vs Top20 Sannolikhet',
        labels={'super_score': 'Super Score', 'Top20_prob': 'Top20 Sannolikhet'}
    )
    
    # 3. Total Strokes Gained
    if 'SG: TOT' in df.columns:
        sg_fig = px.bar(
            df.sort_values('SG: TOT', ascending=False).head(20),
            x='Spelare',
            y='SG: TOT',
            title='Total Strokes Gained (SG: TOT) för Topp 20 Spelare',
            labels={'SG: TOT': 'Total Strokes Gained'}
        )
    else:
        sg_fig = px.bar(title='Ingen Strokes Gained-data tillgänglig')
    
    # 4. NY: Samband mellan features
    correlation_fig = px.scatter(
        df,
        x='FedEx_Points',
        y='SG: TOT',
        color='Top20',
        trendline='ols',
        hover_name='Spelare',
        title='Samband mellan FedEx-poäng och Strokes Gained',
        labels={'FedEx_Points': 'FedEx Poäng', 'SG: TOT': 'Total Strokes Gained'}
    )
    
    # 5. NY: Korrelationsmatris (heatmap)
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr().round(2)
    heatmap_fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title='Korrelation mellan variabler',
        color_continuous_scale='RdBu',
        range_color=[-1, 1]
    )
    
    return hist_fig, scatter_fig, sg_fig, correlation_fig, heatmap_fig


# Uppdatera denna rad:
score_comparison_fig, hist_fig, scatter_fig, correlation_fig = create_visualizations(result_data)

# Och uppdatera app.layout för att inkludera correlation_fig:
app.layout = html.Div([
    html.H1("Golf Prestanda Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Graph(figure=score_comparison_fig)
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(figure=hist_fig)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=scatter_fig)
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(figure=correlation_fig)
    ]),
    
    html.H2("Topp 20 Spelare", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='top-players-table',
        columns=[
            {'name': 'Spelare', 'id': 'Spelare'},
            {'name': 'Oakmont Score', 'id': 'match_score_oakmont', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'PGA Score', 'id': 'match_score_pga', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Top20 Sannolikhet', 'id': 'Top20_prob', 'type': 'numeric', 'format': {'specifier': '.1%'}}
        ],
        data=result_data.head(20).to_dict('records'),
        style_table={'width': '90%', 'margin': '0 auto'},
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white'}
    )
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)


#%%


import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =============================================
# HÄMTA OCH BEARBETA DATA (FRÅN DIN ORIGINALKOD)
# =============================================

def load_and_process_data():
    # Läs in data från HTML-filer
    df_list = pd.read_html("golfprojekt_strokes_gained.html")
    df_fedex_list = pd.read_html("fedex_top100.html")
    pga_df_list = pd.read_html("pga_championship.html")
    
    df = df_list[0]
    df_fedex = df_fedex_list[0]
    pga_df = pga_df_list[0]
    
    # Bearbeta data
    common_players = set(df['Spelare']) & set(df_fedex['Spelare']) & set(pga_df['Spelare'])
    
    df_sg = df[["Spelare", "SG: OTT", "SG: APP", "SG: ATG", "SG: P", "SG: TOT"]].copy()
    df_sg = df_sg[df_sg['Spelare'].isin(common_players)]
    
    for col in df_sg.columns[1:]:
        df_sg.loc[:, col] = pd.to_numeric(df_sg[col], errors="coerce")
    
    df_sg = df_sg.dropna().sort_values(by='SG: TOT', ascending=False)
    
    # Beräkna matchpoäng
    weights = {'SG: OTT': 0.8, 'SG: APP': 0.6, 'SG: ATG': 0.3, 'SG: P': 0.5}
    df_with_scores = calculate_match_scores(df, weights)
    df_with_scores = df_with_scores[df_with_scores['Spelare'].isin(common_players)]
    
    # Slå ihop data
    final_df = pd.merge(
        df_with_scores[['Spelare', 'match_score']],
        df_sg[['Spelare', 'SG: TOT']],
        on='Spelare',
        how='inner'
    )
    
    final_df = pd.merge(
        final_df,
        df_fedex[df_fedex['Spelare'].isin(common_players)][['Spelare', 'FedEx_Points']],
        on='Spelare',
        how='inner'
    )
    
    final_df = pd.merge(
        final_df,
        pga_df[['Spelare', 'Position', 'Top20']],
        on='Spelare',
        how='inner'
    )
    
    return final_df

def calculate_match_scores(df, weights):
    cols = list(weights.keys())
    df_clean = df.dropna(subset=cols).copy()
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['match_score'] = sum(df_clean[col] * weight for col, weight in weights.items())
    return df_clean

def train_model(data):
    # Beräkna super_score
    data['super_score'] = data.apply(calculate_player_score, axis=1)
    min_score, max_score = data['super_score'].min(), data['super_score'].max()
    data['super_score'] = 1 + ((data['super_score'] - min_score) / (max_score - min_score)) * 9
    
    # Rensa data
    cleaned_data = data[
        ~((data['super_score'] > 7) & (data['Position'] > 40)) & 
        ~((data['super_score'] < 3) & (data['Position'] <= 20))
    ].copy()
    
    # Träna modell
    X = cleaned_data[['super_score']]
    y = cleaned_data['Top20']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(
        class_weight='balanced',
        penalty='l2',
        C=0.1,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    cleaned_data['Top20_prob'] = model.predict_proba(X)[:, 1]
    cleaned_data['Top20_prob'] = calculate_probabilities(cleaned_data['Top20_prob'])
    
    return cleaned_data.sort_values('Top20_prob', ascending=False)

def calculate_player_score(player):
    fedex_normalized = (player['FedEx_Points'] - 250) / (3800 - 250)
    return 0.4 * player['match_score'] + 0.4 * player['SG: TOT'] + 0.2 * fedex_normalized

def calculate_probabilities(probability_series):
    raw_prob = np.clip(probability_series, 0.01, 0.99)
    scaled_prob = 0.05 + 0.90 * (raw_prob - np.min(raw_prob)) / (np.max(raw_prob) - np.min(raw_prob) + 1e-10)
    return np.clip(scaled_prob, 0.05, 0.95)
app = dash.Dash(__name__)

# Ladda och processa data
raw_data = load_and_process_data()
result_data = train_model(raw_data)

# Skapa visualiseringar
def create_visualizations(df):
    # 1. Histogram över super_score
    hist_fig = px.histogram(
        df, 
        x='super_score', 
        color='Top20',
        nbins=20,
        title='Fördelning av Super Score',
        labels={'super_score': 'Super Score', 'count': 'Antal spelare'}
    )
    
    # 2. Spridningsdiagram för Top20 prediktioner
    scatter_fig = px.scatter(
        df,
        x='super_score',
        y='Top20_prob',
        color='Position',
        size='FedEx_Points',
        hover_name='Spelare',
        title='Super Score vs Top20 Sannolikhet',
        labels={'super_score': 'Super Score', 'Top20_prob': 'Top20 Sannolikhet'}
    )
    
    # 3. Total Strokes Gained
    if 'SG: TOT' in df.columns:
        sg_fig = px.bar(
            df.sort_values('SG: TOT', ascending=False).head(20),
            x='Spelare',
            y='SG: TOT',
            title='Total Strokes Gained (SG: TOT) för Topp 20 Spelare',
            labels={'SG: TOT': 'Total Strokes Gained'}
        )
    else:
        sg_fig = px.bar(title='Ingen Strokes Gained-data tillgänglig')
    
    # 4. Samband mellan features
    correlation_fig = px.scatter(
        df,
        x='FedEx_Points',
        y='SG: TOT',
        color='Top20',
        trendline='ols',
        hover_name='Spelare',
        title='Samband mellan FedEx-poäng och Strokes Gained',
        labels={'FedEx_Points': 'FedEx Poäng', 'SG: TOT': 'Total Strokes Gained'}
    )
    
    # 5. Korrelationsmatris (heatmap)
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr().round(2)
    heatmap_fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title='Korrelation mellan variabler',
        color_continuous_scale='RdBu',
        range_color=[-1, 1]
    )
    
    return hist_fig, scatter_fig, sg_fig, correlation_fig, heatmap_fig

# Hämta alla 5 figurer
hist_fig, scatter_fig, sg_fig, correlation_fig, heatmap_fig = create_visualizations(result_data)

# Uppdaterad layout som inkluderar alla figurer
app.layout = html.Div([
    html.H1("Golf Prestanda Dashboard", style={'textAlign': 'center'}),
    
    # Första raden med histogram och scatter
    html.Div([
        html.Div([
            dcc.Graph(figure=hist_fig)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=scatter_fig)
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    # Andra raden med SG-figur och korrelation
    html.Div([
        html.Div([
            dcc.Graph(figure=sg_fig)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=correlation_fig)
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    # Tredje raden med heatmap
    html.Div([
        dcc.Graph(figure=heatmap_fig)
    ], style={'width': '80%', 'margin': '0 auto'}),
    
    # Tabell med toppspelare
    html.H2("Topp 20 Spelare", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='top-players-table',
        columns=[
            {'name': 'Spelare', 'id': 'Spelare'},
            {'name': 'Match Score', 'id': 'match_score', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            #{'name': 'PGA Score', 'id': 'match_score_pga', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Top20 Sannolikhet', 'id': 'Top20_prob', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            {'name': 'SG: TOT', 'id': 'SG: TOT', 'type': 'numeric', 'format': {'specifier': '.2f'}}
        ],
        data=result_data.head(20).to_dict('records'),
        style_table={'width': '90%', 'margin': '0 auto'},
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white'}
    )
])

if __name__ == '__main__':
    app.run(debug=True, port=8050)



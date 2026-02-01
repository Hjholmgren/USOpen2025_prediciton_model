#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 01:21:51 2025

@author: henrik

Jag har skapat en linjär regressionsmodell för att prediktera top 20 på US Open i golf.
Jag ahr tagit värden från spelarnas Shots Gained, viktat Shots Gained för vad som anses fördelaktigt
på Oakmont banan och tagit data från top 100 i FedEx. Modellen har jag sedan tränat mot resultatet
från PGA Championship för en månad sedan och därefter tillämpat den på US Open startfältet. Då 
datan inte gick att web scrapea har jag gjort egna html filer. 
"""



    
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =============================================
# DATAINLÄSNING OCH BEARBETNING
# =============================================

def load_and_process_data():
    """
    Laddar in och bearbetar data från tre olika källor:
    1. Strokes gained-statistik från US Open
    2. FedEx Cup-ranking
    3. PGA Championship-resultat
    
    Returnerar en sammanslagen DataFrame med gemensamma spelare
    """
    # Läs in data från HTML-filer
    df_list = pd.read_html("golfprojekt_strokes_gained.html")  # US Open-data
    df_fedex_list = pd.read_html("fedex_top100.html")          # FedEx-data
    pga_df_list = pd.read_html("pga_championship.html")        # PGA-data
    
    # Extrahera första tabellen från varje fil
    df = df_list[0]
    df_fedex = df_fedex_list[0]
    pga_df = pga_df_list[0]
    
    # Hitta spelare som deltagit i alla tre turneringar
    common_players = set(df['Spelare']) & set(df_fedex['Spelare']) & set(pga_df['Spelare'])
    
    # Extrahera och rensa strokes gained-data
    df_sg = df[["Spelare", "SG: OTT", "SG: APP", "SG: ATG", "SG: P", "SG: TOT"]].copy()
    df_sg = df_sg[df_sg['Spelare'].isin(common_players)]  # Behåller endast gemensamma spelare
    
    # Konvertera till numeriska värden
    for col in df_sg.columns[1:]:
        df_sg.loc[:, col] = pd.to_numeric(df_sg[col], errors="coerce")
    
    # Ta bort rader med saknade värden och sortera
    df_sg = df_sg.dropna().sort_values(by='SG: TOT', ascending=False)
    
    # Beräkna viktade matchpoäng för Oakmont
    weights = {'SG: OTT': 0.8, 'SG: APP': 0.6, 'SG: ATG': 0.3, 'SG: P': 0.5}
    df_with_scores = calculate_match_scores(df, weights)
    df_with_scores = df_with_scores[df_with_scores['Spelare'].isin(common_players)]
    
    # Slå ihop all data till en gemensam DataFrame
    final_df = pd.merge(
        df_with_scores[['Spelare', 'match_score']],
        df_sg[['Spelare', 'SG: TOT']],
        on='Spelare',
        how='inner'
    )
    
    # Lägg till FedEx-poäng
    final_df = pd.merge(
        final_df,
        df_fedex[df_fedex['Spelare'].isin(common_players)][['Spelare', 'FedEx_Points']],
        on='Spelare',
        how='inner'
    )
    
    # Lägg till PGA-resultat (position och Top20-markör)
    final_df = pd.merge(
        final_df,
        pga_df[['Spelare', 'Position', 'Top20']],
        on='Spelare',
        how='inner'
    )
    
    return final_df

def calculate_match_scores(df, weights):
    """
    Beräknar viktade SG baserat på Oakmont banan
    
    """
    cols = list(weights.keys())
    # Ta bort rader med saknade värden i relevanta kolumner
    df_clean = df.dropna(subset=cols).copy()
    
    # Konvertera till numeriska värden
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Beräkna viktat medelvärde
    df_clean['match_score'] = sum(df_clean[col] * weight for col, weight in weights.items())
    return df_clean


def train_model(data):
    """
    Tränar en logistisk regressionsmodell för att förutsäga Top20-sannolikhet
    
    Steg:
    1. Beräknar super_score (kombinerad poäng)
    2. Rensar data från orimliga kombinationer
    3. Tränar modellen
    4. Beräknar predikterade sannolikheter
    
    Returnerar:
        DataFrame med tillagda kolumner för super_score och Top20_prob
    """
    # Beräkna super_score (kombination av match_score, SG: TOT och FedEx-poäng)
    data['super_score'] = data.apply(calculate_player_score, axis=1)
    
    # Skala om till 1-10 skala
    min_score, max_score = data['super_score'].min(), data['super_score'].max()
    data['super_score'] = 1 + ((data['super_score'] - min_score) / (max_score - min_score)) * 9
    
    # Rensa data - ta bort orimliga kombinationer
    cleaned_data = data[
        ~((data['super_score'] > 7) & (data['Position'] > 40)) &  # Hög poäng men dålig placering
        ~((data['super_score'] < 3) & (data['Position'] <= 20))   # Låg poäng men bra placering
    ].copy()
    
    # Förbered data för modellträning
    X = cleaned_data[['super_score']]  # Förklarande variabel
    y = cleaned_data['Top20']          # Målvariabel (1 om Top20, 0 annars)
    
    # Dela upp i tränings- och testdata
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42  # För reproducerbara resultat
    )
    
    # Skapa och träna modell
    model = LogisticRegression(
        class_weight='balanced',  # Balansera klasserna
        penalty='l2',            # Regularisering
        C=0.1,                  # Regulariseringsstyrka
        max_iter=1000,          # Max antal iterationer
        random_state=42         # För reproducerbara resultat
    )
    model.fit(X_train, y_train)
    
    # Gör prediktioner på hela datasetet
    cleaned_data['Top20_prob'] = model.predict_proba(X)[:, 1]
    
    # Skala sannolikheterna till 5-95% intervall
    cleaned_data['Top20_prob'] = calculate_probabilities(cleaned_data['Top20_prob'])
    
    # Sortera efter sannolikhet
    return cleaned_data.sort_values('Top20_prob', ascending=False)

def calculate_player_score(player):
    """
    Beräknar en sammansatt prestationspoäng (super_score) för en spelare

    """
    # Normalisera FedEx-poängen till 0-1 intervall
    fedex_normalized = (player['FedEx_Points'] - 250) / (3800 - 250)
    
    # Kombinera komponenterna med vikter
    return 0.4 * player['match_score'] + 0.4 * player['SG: TOT'] + fedex_normalized

def calculate_probabilities(probability_series):
    """
    Skalar sannolikheter till 5-95% intervall
    
    """
    # Begränsa till 1-99% för att undvika extremvärden
    raw_prob = np.clip(probability_series, 0.01, 0.99)
    
    # Linjär skalning till 5-95% intervall
    scaled_prob = 0.05 + 0.90 * (raw_prob - np.min(raw_prob)) / (np.max(raw_prob) - np.min(raw_prob) + 1e-10)
    
    # Ytterligare säkerhetsklämma
    return np.clip(scaled_prob, 0.05, 0.95)

# =============================================
# DASH-APPLIKATION
# =============================================

# Initiera Dash-app
app = dash.Dash(__name__)

# Ladda och processa data
raw_data = load_and_process_data()
result_data = train_model(raw_data)

def create_visualizations(df):
    """
    Skapar visualiseringar för dashboarden
    
    Returnerar:
        5 Plotly-figurer:
        1. Histogram över super_score
        2. Spridningsdiagram för Top20-sannolikhet
        3. Staplad för Total Strokes Gained
        4. Samband mellan FedEx-poäng och SG: TOT
        5. Korrelationsmatris
    """
    # 1. Histogram över super_score-fördelning
    hist_fig = px.histogram(
        df, 
        x='super_score', 
        color='Top20',  # Färgkodad efter Top20-status
        nbins=20,
        title='Fördelning av Super Score',
        labels={'super_score': 'Super Score', 'count': 'Antal spelare'}
    )
    
    # 2. Spridningsdiagram: Super Score vs Top20-sannolikhet
    scatter_fig = px.scatter(
        df,
        x='super_score',
        y='Top20_prob',
        color='Position',       # Färg efter slutposition
        size='FedEx_Points',   # Storlek efter FedEx-poäng
        hover_name='Spelare',  # Visa namn vid hover
        title='Super Score vs Top20 Sannolikhet',
        labels={
            'super_score': 'Super Score', 
            'Top20_prob': 'Top20 Sannolikhet'
        }
    )
    
    # 3. Total Strokes Gained för topp 20 spelare
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
    
    # 4. Samband mellan FedEx-poäng och Strokes Gained
    correlation_fig = px.scatter(
        df,
        x='FedEx_Points',
        y='SG: TOT',
        color='Top20',        # Färgkodad efter Top20-status
        trendline='ols',      # Linjär regression
        hover_name='Spelare',
        title='Samband mellan FedEx-poäng och Strokes Gained',
        labels={
            'FedEx_Points': 'FedEx Poäng', 
            'SG: TOT': 'Total Strokes Gained'
        }
    )
    
    # 5. Korrelationsmatris mellan numeriska variabler
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_cols].corr().round(2)
    heatmap_fig = px.imshow(
        corr_matrix,
        text_auto=True,                  # Visa värden i celler
        title='Korrelation mellan variabler',
        color_continuous_scale='RdBu',   # Röd-blå färgskala
        range_color=[-1, 1]              # Fast intervall för korrelation
    )
    
    return hist_fig, scatter_fig, sg_fig, correlation_fig, heatmap_fig

# Skapa alla visualiseringar
hist_fig, scatter_fig, sg_fig, correlation_fig, heatmap_fig = create_visualizations(result_data)

# Definiera layout för dashboard
app.layout = html.Div([
    # Rubrik
    html.H1("US Open Predicition Dashboard", style={'textAlign': 'center'}),
    
    # Första raden: Histogram och spridningsdiagram
    html.Div([
        html.Div([
            dcc.Graph(figure=hist_fig)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=scatter_fig)
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    # Andra raden: Strokes gained och korrelationsdiagram
    html.Div([
        html.Div([
            dcc.Graph(figure=sg_fig)
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=correlation_fig)
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    
    # Tredje raden: Korrelationsmatris (centrerad)
    html.Div([
        dcc.Graph(figure=heatmap_fig)
    ], style={'width': '80%', 'margin': '0 auto'}),
    
    # Tabell med topp 20 spelare
    html.H2("Topp 20 Spelare", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='top-players-table',
        columns=[
            {'name': 'Spelare', 'id': 'Spelare'},
            {'name': 'Viktad SG Oakmont', 'id': 'match_score', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Top20 Sannolikhet', 'id': 'Top20_prob', 'type': 'numeric', 'format': {'specifier': '.1%'}},
            {'name': 'SG: TOT', 'id': 'SG: TOT', 'type': 'numeric', 'format': {'specifier': '.2f'}}
        ],
        data=result_data.head(20).to_dict('records'),
        style_table={'width': '90%', 'margin': '0 auto'},
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
])

if __name__ == '__main__':
    
    
    # Starta applikationen
    app.run(debug=True, port=8050)
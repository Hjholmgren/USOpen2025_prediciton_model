#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:01:12 2025

@author: henrik
"""

import pandas as pd

def calculate_match_scores(df, weights):
    cols = list(weights.keys())
    df_clean = df.dropna(subset=cols).copy()
    
    # Säkerställ att kolumner är numeriska
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Räkna match_score som viktad summa
    df_clean['match_score'] = sum(df_clean[col] * weight for col, weight in weights.items())
    return df_clean
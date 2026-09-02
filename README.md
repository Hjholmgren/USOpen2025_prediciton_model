# US Open 2025 Top-20 Predictor

A statistical model that estimates each golfer's probability of finishing
**top 20** at the 2025 US Open (Oakmont), built from Strokes Gained data,
FedEx Cup standings, and recent PGA Championship results. Results are shown
in an interactive Dash dashboard.

## How it works

1. **Combine three data sources** for players who appear in all of them:
   - Strokes Gained stats (Off-the-Tee, Approach, Around-the-Green, Putting, Total)
   - FedEx Cup points (top 100)
   - PGA Championship finishing position (used as the training label — a
     recent, similar-quality field to validate against)

2. **Build a course-fit score (`match_score`)** — a weighted sum of Strokes
   Gained categories, weighted toward the skills that matter most on
   Oakmont's setup:

   | Category | Weight |
   |---|---|
   | SG: Off-the-Tee | 0.8 |
   | SG: Approach | 0.6 |
   | SG: Around-the-Green | 0.3 |
   | SG: Putting | 0.5 |

3. **Combine into a `super_score`** — `match_score`, total Strokes Gained, and
   normalized FedEx Cup points blended (40% / 40% / 20%) and rescaled to a
   1–10 range.

4. **Train a logistic regression** on `super_score` against actual PGA
   Championship top-20 finishes, with balanced class weights and L2
   regularization. Obviously unreliable training rows (e.g. a high score
   paired with a poor finish) are filtered out first.

5. **Predict top-20 probability** for the US Open field, scaled to a 5–95%
   range to avoid overconfident 0%/100% outputs.

## Dashboard

Running the script launches a Dash app with:

- Histogram of `super_score`, colored by top-20 outcome
- Scatter of `super_score` vs predicted top-20 probability
- Bar chart of total Strokes Gained for the top 20 players
- FedEx points vs Strokes Gained, with an OLS trendline
- Correlation heatmap across all numeric features
- A sortable table of the top 20 predicted players, with probability
  highlighted when it clears 70%

## Run locally

```bash
pip install dash plotly pandas numpy scikit-learn
python "US_Open_Prediction .py"
```

Then open <http://localhost:8050> in your browser.

## Project structure

```
US_Open_Prediction .py         # main script — data pipeline, model, dashboard
match_score.py                 # standalone match_score calculation
random_forest_golfprojection.py# exploratory / scratch versions of the model (not the final pipeline)
golfprojekt_strokes_gained.html# US Open field: Strokes Gained stats
fedex_top100.html              # FedEx Cup top 100 rankings
pga_championship.html          # PGA Championship results (training data)
players_in_all_three_events.csv# players common to all three sources
us_open_top20_predictions.csv  # model output
```

## Data notes

The source stats weren't available to scrape programmatically, so the three
HTML files are manual snapshots taken shortly before the 2025 US Open rather
than a live feed. To refresh predictions for a different event, replace
those three files with current data in the same table format.

`random_forest_golfprojection.py`, despite the name, doesn't currently train
a random forest — it's an earlier, iterative exploration of the logistic
regression approach (including a version with simulated data) and is kept
for reference rather than as the primary pipeline.

## Disclaimer

Built as a personal/portfolio data science project. Small sample sizes
(one training tournament) and course-fit weights based on judgment rather
than optimization mean this shouldn't be treated as a serious betting or
forecasting tool.

## Possible extensions

- Train on multiple past tournaments instead of one, for a larger and more
  robust training set
- Automate data collection instead of manual HTML snapshots
- Try a proper random forest or gradient-boosted model and compare against
  the logistic regression baseline
- Add course-history and recent-form features beyond Strokes Gained





<img width="1375" height="496" alt="Skärmavbild 2025-06-15 kl  01 37 01" src="https://github.com/user-attachments/assets/f473ed7f-97ce-40b6-b677-c77712da4896" />
<img width="1102" height="432" alt="Skärmavbild 2025-06-15 kl  01 37 18" src="https://github.com/user-attachments/assets/6509cdd5-ce0b-4845-bb69-a77115ab38e8" />
<img width="1102" height="432" alt="Skärmavbild 2025-06-15 kl  01 37 18" src="https://github.com/user-attachments/assets/63ed2808-d336-484c-b5fa-e3768e44e892" />
<img width="1362" height="518" alt="Skärmavbild 2025-06-15 kl  01 38 02" src="https://github.com/user-attachments/assets/a2b7c569-9e73-4a56-acee-2cb0e0663ea1" />
<img width="1361" height="355" alt="Skärmavbild 2025-06-15 kl  01 38 14" src="https://github.com/user-attachments/assets/97725e2d-1433-42f0-a637-96f23c18305c" />

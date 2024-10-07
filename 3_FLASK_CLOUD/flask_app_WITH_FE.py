from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the saved model, feature names, scaler, and decision threshold
model = load('/home/Isdinval/mysite/nba_career_prediction_model_WITH_FE.joblib')
feature_names = load('/home/Isdinval/mysite/feature_names_WITH_FE.joblib')
scaler = load('/home/Isdinval/mysite/scaler_WITH_FE.joblib')

with open('/home/Isdinval/mysite/decision_threshold_WITH_FE.txt', 'r') as f:
    decision_threshold = float(f.read().strip())

@app.route('/')
def home():
    return render_template('index_WITH_FE.html')  # Renders the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (coming from the form)
    data = request.form

    # Ensure that the base features are present
    required_base_features = {'GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV' }

    provided_features = set(data.keys())

    if not required_base_features.issubset(provided_features):
        missing_features = required_base_features - provided_features
        return jsonify({'error': f'Missing base features: {", ".join(missing_features)}'}), 400

    # Extract values for the base features in the correct order
    gp = float(data['GP'])
    min_played = float(data['MIN'])
    pts = float(data['PTS'])
    fgm = float(data['FGM'])
    fga = float(data['FGA'])
    fg_percentage = float(data['FG%'])
    threes_made = float(data['3P Made'])
    threes_attempted = float(data['3PA'])
    threes_percentage = float(data['3P%'])
    ftm = float(data['FTM'])
    fta = float(data['FTA'])
    ft_percentage = float(data['FT%'])
    oreb = float(data['OREB'])
    dreb = float(data['DREB'])
    reb = float(data['REB'])
    ast = float(data['AST'])
    stl = float(data['STL'])
    blk = float(data['BLK'])
    tov = float(data['TOV'])


    # Compute the new features

    # Efficiency Metrics
    PER = (pts + reb + ast + stl + blk - tov) / gp
    PPM = pts / min_played
    Usage_Rate = (fga + 0.44 * fta + tov) / min_played

    # Scoring Efficiency
    True_Shooting_Percentage = pts / (2 * (fga + 0.44 * fta))
    Effective_FG_Percentage = (fgm + 0.5 * threes_made) / fga
    Points_Per_Shot = pts / (fga + 0.44 * fta)

    # Rebounding
    Total_Rebound_Percentage = reb / (gp * min_played / 48)
    Offensive_Rebound_Percentage = oreb / (gp * min_played / 48)
    Defensive_Rebound_Percentage = dreb / (gp * min_played / 48)

    # Passing and Ball Handling
    Assist_Percentage = ast / (min_played / 48 * gp * 5)
    assist_turnover_ratio = ast / (tov if tov != 0 else 1)
    assist_ratio = ast * 100 / fga

    # Defensive Metrics
    Steal_Percentage = stl / (gp * min_played / 48)
    Block_Percentage = blk / (gp * min_played / 48)
    Stocks = stl + blk

    # Versatility Metrics
    Versatility_Index = pts + reb + ast + stl + blk
    Offensive_Versatility = pts + ast + oreb
    Defensive_Versatility = dreb + stl + blk

    # Shooting Metrics
    Three_Point_Rate = threes_attempted / fga
    Free_Throw_Rate = fta / fga

    # Per Game and Per 36 Minutes Metrics
    PTS_Per_Game = pts / gp
    REB_Per_Game = reb / gp
    AST_Per_Game = ast / gp
    STL_Per_Game = stl / gp
    BLK_Per_Game = blk / gp
    TOV_Per_Game = tov / gp

    PTS_Per_36 = pts / min_played * 36
    REB_Per_36 = reb / min_played * 36
    AST_Per_36 = ast / min_played * 36
    STL_Per_36 = stl / min_played * 36
    BLK_Per_36 = blk / min_played * 36
    TOV_Per_36 = tov / min_played * 36

    # Advanced Metrics
    Box_Plus_Minus = (pts + 0.2 * oreb + 0.8 * dreb + 2.7 * ast + stl + 0.7 * blk - 0.7 * fga - 0.4 * (fta - ftm) - 1.2 * tov) / gp

    # Consistency Metrics
    Scoring_Consistency = pts / (fga + fta)
    Usage_Consistency = Usage_Rate / gp

    # Compound Metrics
    Offensive_Impact = (pts + ast) / min_played
    Defensive_Impact = (dreb + stl + blk) / min_played
    Overall_Impact = Offensive_Impact + Defensive_Impact

    # Efficiency Ratios
    Points_Per_Touch = pts / (fga + fta + ast + tov)
    Production_Per_Possession = (pts + ast + oreb) / (fga - oreb + tov + 0.44 * fta)

    # Relative Performance Metrics
    PTS_to_Usage_Ratio = pts / Usage_Rate
    AST_to_Usage_Ratio = ast / Usage_Rate

    # Shooting Splits
    Two_Point_Percentage = (fgm - threes_made) / (fga - threes_attempted)
    Points_Per_Shot_Attempt = pts / fga

    # Physical Impact Metrics
    Physical_Impact = (reb + blk + stl) / min_played

    # Add these new features to a vector for prediction
    feature_vector = [gp, min_played, pts, fgm, fga, fg_percentage, threes_made, threes_attempted,
        threes_percentage, ftm, fta, ft_percentage, oreb, dreb, reb, ast, stl, blk, tov,
        PER, PPM, Usage_Rate, True_Shooting_Percentage, Effective_FG_Percentage, Points_Per_Shot,
        Total_Rebound_Percentage, Offensive_Rebound_Percentage, Defensive_Rebound_Percentage,
        Assist_Percentage, assist_turnover_ratio, Steal_Percentage, Block_Percentage, Stocks,
        Versatility_Index, Offensive_Versatility, Defensive_Versatility,
        Three_Point_Rate, Free_Throw_Rate,
        PTS_Per_Game, PTS_Per_36, REB_Per_Game, REB_Per_36, AST_Per_Game, AST_Per_36,
        STL_Per_Game, STL_Per_36, BLK_Per_Game, BLK_Per_36, TOV_Per_Game, TOV_Per_36,
        Box_Plus_Minus, Scoring_Consistency, Usage_Consistency, Offensive_Impact,
        Defensive_Impact, Overall_Impact, Points_Per_Touch, Production_Per_Possession,
        PTS_to_Usage_Ratio, AST_to_Usage_Ratio,
        Two_Point_Percentage, Points_Per_Shot_Attempt, Physical_Impact
    ]
    
    # Preprocess the data (scaling)
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)

    # Make a prediction with the model
    probability = model.predict_proba(feature_vector)[0][1]
    prediction = 1 if probability >= decision_threshold else 0

    # Prepare the response
    result = {
        'prediction': prediction,
        'probability': float(probability),
        'interpretation': 'Likely to last 5+ years in NBA' if prediction == 1 else 'Likely to last <5 years in NBA',
        'threshold_used': decision_threshold
    }

    return render_template('result.html', result=result)

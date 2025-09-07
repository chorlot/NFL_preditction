import streamlit as st  # ✅ Importer Streamlit en premier

# ✅ Configurer la page immédiatement après l'import
st.set_page_config(
    page_title="NFL Match Predictor",
    page_icon="🏈",
    layout="wide"
)

import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# ✅ Vérifie que rien ne précède `st.set_page_config()` (aucun affichage ou chargement de données ici)


TEAM_ALIASES = {
    "Arizona Cardinals": "ARI", "Arizona": "ARI", "Cardinals": "ARI",
    "Atlanta Falcons": "ATL", "Atlanta": "ATL", "Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Baltimore": "BAL", "Ravens": "BAL",
    "Buffalo Bills": "BUF", "Buffalo": "BUF", "Bills": "BUF",
    "Carolina Panthers": "CAR", "Carolina": "CAR", "Panthers": "CAR",
    "Chicago Bears": "CHI", "Chicago": "CHI", "Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cincinnati": "CIN", "Bengals": "CIN",
    "Cleveland Browns": "CLE", "Cleveland": "CLE", "Browns": "CLE",
    "Dallas Cowboys": "DAL", "Dallas": "DAL", "Cowboys": "DAL",
    "Denver Broncos": "DEN", "Denver": "DEN", "Broncos": "DEN",
    "Detroit Lions": "DET", "Detroit": "DET", "Lions": "DET",
    "Green Bay Packers": "GB", "Green Bay": "GB", "Packers": "GB",
    "Houston Texans": "HOU", "Houston": "HOU", "Texans": "HOU",
    "Indianapolis Colts": "IND", "Indianapolis": "IND", "Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Jacksonville": "JAX", "Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Kansas City": "KC", "Kansas": "KC", "Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Las Vegas": "LV", "Raiders": "LV",
    "Los Angeles Chargers": "LAC", "LA Chargers": "LAC", "Chargers": "LAC",
    "Los Angeles Rams": "LAR", "LA Rams": "LAR", "Rams": "LAR",
    "Miami Dolphins": "MIA", "Miami": "MIA", "Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "Minnesota": "MIN", "Vikings": "MIN",
    "New England Patriots": "NE", "New England": "NE", "Patriots": "NE",
    "New Orleans Saints": "NO", "New Orleans": "NO", "Saints": "NO",
    "New York Giants": "NYG", "Giants": "NYG",
    "New York Jets": "NYJ", "Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Philadelphia": "PHI", "Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "Pittsburgh": "PIT", "Steelers": "PIT",
    "San Francisco 49ers": "SF", "San Francisco": "SF", "49ers": "SF",
    "Seattle Seahawks": "SEA", "Seattle": "SEA", "Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tampa Bay": "TB", "Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Tennessee": "TEN", "Titans": "TEN",
    "Washington Commanders": "WAS", 
    }

# Dictionnaire inverse pour retrouver le nom complet d'une équipe à partir des abréviations
TEAM_FULL_NAMES = {abbr: full_name for full_name, abbr in TEAM_ALIASES.items()}


def normalize_team_name(team_input):
    """
    Cette fonction prend en entrée un texte (abréviation, ville, nom d'équipe, ou combinaison)
    et renvoie l'abréviation normalisée selon le dictionnaire TEAM_ALIASES.
    """
    if not team_input:
        return None
    # On normalise en supprimant les espaces superflus et en passant en minuscules pour la comparaison
    input_clean = team_input.strip().lower()
    
    # D'abord, si l'entrée correspond déjà à une abréviation connue (en insensibilité de casse)
    for abbr in TEAM_ALIASES.values():
        if input_clean == abbr.lower():
            return abbr  # On retourne l'abréviation en majuscules
    
    # Ensuite, on teste si l'entrée apparaît dans le nom complet (ou partiel)
    # Par exemple, "washington", "commanders", etc.
    for full_name, abbr in TEAM_ALIASES.items():
        if input_clean in full_name.lower():
            return abbr
    
    # Si aucune correspondance n'est trouvée, on retourne None
    return None


# ========== 🔄 OPTIMISATION DU CHARGEMENT DES DONNÉES ==========
@st.cache_data
def load_data(start_year=2020, end_year=2024):
    """Charge les données des matchs et statistiques des équipes."""
    games = nfl.import_schedules(list(range(start_year, end_year + 1)))
    games = games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']]
    games['winner_team'] = games.apply(lambda row: row['home_team'] if row['home_score'] > row['away_score'] else row['away_team'], axis=1)
    games.dropna(subset=['winner_team'], inplace=True)
    
    player_stats = nfl.import_weekly_data(list(range(start_year, end_year + 1)))
    team_stats = player_stats.groupby(['season', 'week', 'recent_team']).mean().reset_index()
    return games, team_stats

games, team_stats = load_data()

# ========== 🔄 CONSTRUCTION DU DATASET ==========
@st.cache_data
def prepare_dataset(games, team_stats):
    """Construit le dataset pour l'entraînement du modèle."""
    df = games.merge(team_stats, left_on=['season', 'week', 'home_team'], right_on=['season', 'week', 'recent_team'])
    df = df.merge(team_stats, left_on=['season', 'week', 'away_team'], right_on=['season', 'week', 'recent_team'], suffixes=('_home', '_away'))
    df['winner'] = (df['winner_team'] == df['home_team']).astype(int)
    df.drop(columns=['recent_team_home', 'recent_team_away', 'winner_team'], inplace=True)
    return df

df = prepare_dataset(games, team_stats)

# ========== 🔄 DIVISION DES DONNÉES : ENTRAÎNEMENT AVANT 2024, TEST SUR 2024 ==========
train_data = df[df['season'] < 2024]
test_data = df[df['season'] == 2024]

features = [col for col in df.columns if '_home' in col or '_away' in col and col not in ['season', 'week']]
X_train = train_data[features]
y_train = train_data['winner']
X_test = test_data[features]
y_test = test_data['winner']

# ========== 🔄 ENTRAÎNEMENT DU MODÈLE ==========


# Vérifier si certaines équipes dominent le dataset (ex: Kansas City, Buffalo)
max_games_per_team = 50  # Limite arbitraire, ajustable

df_limited = df.groupby("home_team").apply(lambda x: x.sample(min(len(x), max_games_per_team), random_state=42))
df_limited = df_limited.groupby("away_team").apply(lambda x: x.sample(min(len(x), max_games_per_team), random_state=42))

df_limited.reset_index(drop=True, inplace=True)

# Sélection des features et de la cible
# Version réduite des features pour éviter le sur-ajustement
features = ["completions_home", "attempts_home", "passing_yards_home",
            "rushing_yards_home", "completions_away", "attempts_away",
            "passing_yards_away", "rushing_yards_away", "sacks_home", "sacks_away"]

# Calcul du taux de victoires à domicile et défaites en extérieur *sans inclure la saison actuelle*
latest_season = df["season"].max()

home_wins = df[df["season"] < latest_season].groupby("home_team")["winner"].mean().rename("home_win_rate")
away_wins = df[df["season"] < latest_season].groupby("away_team")["winner"].apply(lambda x: 1 - x.mean()).rename("away_loss_rate")

# Fusionner ces nouvelles variables avec le dataset
df = df.merge(home_wins, left_on="home_team", right_index=True, how="left")
df = df.merge(away_wins, left_on="away_team", right_index=True, how="left")

# Remplacer les valeurs NaN par la moyenne globale
df["home_win_rate"].fillna(df["home_win_rate"].mean(), inplace=True)
df["away_loss_rate"].fillna(df["away_loss_rate"].mean(), inplace=True)


# Ajout des nouvelles colonnes comme features
# Supprime temporairement les taux de victoires pour voir si le modèle s'appuie trop dessus
features = [col for col in df.columns if ("_home" in col or "_away" in col) and col not in ["season", "week", "home_win_rate", "away_loss_rate"]]

X = df[features]
y = df["winner"]

train_df = df[df["season"] < 2024]  # On entraîne jusqu'en 2023
test_df = df[df["season"] == 2024]  # On teste sur 2024
X_train = train_df[features]
y_train = train_df["winner"]
X_test = test_df[features]
y_test = test_df["winner"]

# ✅ Remplacer les valeurs NaN par la moyenne de la colonne
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Vérifier s'il reste des NaN (au cas où)
print("📊 Vérification après remplissage des NaN :")
print(f"X_train NaN : {X_train.isna().sum().sum()}")
print(f"X_test NaN : {X_test.isna().sum().sum()}")

# ⚠️ Si des NaN persistent, affiche les colonnes concernées
if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
    print("⚠️ Colonnes avec NaN dans X_train :", X_train.columns[X_train.isna().any()].tolist())
    print("⚠️ Colonnes avec NaN dans X_test :", X_test.columns[X_test.isna().any()].tolist())

# Vérifier les données avant l'entraînement
print("📊 Vérification des données avant entraînement :")
print("🔍 X_train (features d'entraînement) :")
print(X_train.head())
print("🔍 y_train (cible d'entraînement) :")
print(y_train.head())
print("🔍 X_test (features de test) :")
print(X_test.head())
print("🔍 y_test (cible de test) :")
print(y_test.head())

# Vérifier si X_test contient des matchs futurs (ce qui serait une erreur !)
print("📆 Année la plus récente utilisée pour le test :", df["season"].max())

# Entraînement du modèle
print("🔄 Entraînement du modèle...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,       # On garde 100 arbres pour une bonne stabilité
    max_depth=3,            # On réduit encore la profondeur pour limiter le surapprentissage
    min_samples_split=30,    # On force le modèle à faire des splits plus larges
    min_samples_leaf=25,     # On évite les feuilles trop spécifiques
    max_features="sqrt",     # On limite encore plus le nombre de variables utilisées par arbre
    random_state=42
)

from sklearn.model_selection import cross_val_score, StratifiedKFold

# Utiliser une validation croisée plus robuste
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='accuracy')

print(f"📊 Score moyen en cross-validation : {cv_scores.mean():.2f}")
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"📊 Moyenne de la cross-validation : {cv_mean:.2f}")
print(f"📊 Écart-type de la cross-validation : {cv_std:.2f}")

print(f"📊 Score minimum : {cv_scores.min():.2f}")
print(f"📊 Score maximum : {cv_scores.max():.2f}")


rf_classifier.fit(X_train, y_train)
# Modèle baseline : toujours prédire "victoire à domicile"
baseline_predictions = [1] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
print(f"📊 Baseline (toujours victoire à domicile) : {baseline_accuracy * 100:.2f}%")

# Modèle aléatoire (50/50)
import numpy as np
random_predictions = np.random.choice([0, 1], size=len(y_test))
random_accuracy = accuracy_score(y_test, random_predictions)
print(f"🎲 Modèle aléatoire (Random Guessing) : {random_accuracy * 100:.2f}%")


from sklearn.metrics import classification_report
# Évaluation du modèle
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Précision du modèle sur l'ensemble de test : {accuracy * 100:.2f}%")
print(f"📊 Précision du modèle : {accuracy * 100:.2f}%")
# Vérifier si le modèle overfit
train_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train))
print(f"📊 Précision sur l'entraînement : {train_accuracy * 100:.2f}%")
print(f"📊 Précision sur le test : {accuracy * 100:.2f}%")

print("📋 Rapport de classification détaillé :")
print(classification_report(y_test, y_pred))  # Voir la précision pour chaque classe

# ========== 🔮 PRÉDICTION DES MATCHS FUTURS ==========

def compute_weighted_team_stats(team_stats, team):
    """Calcule la moyenne pondérée des statistiques des 5 derniers matchs pour une équipe donnée."""
    team_df = team_stats[team_stats["recent_team"] == team].sort_values(by=["season", "week"])
    
    if team_df.empty:
        return None  # ⚠️ Éviter une erreur si l'équipe n'a pas de stats

    # Calcul de la moyenne pondérée
    weighted_stats = team_df.iloc[-5:].ewm(alpha=0.3, adjust=False).mean(numeric_only=True)
    
    # ⚠️ Convertir correctement en dictionnaire de valeurs scalaires
    return weighted_stats.iloc[-1].to_dict()

def predict_future_match(home_team, away_team):
    """Prédit un match futur en utilisant les stats récentes des équipes."""
    print(f"🔮 Prédiction pour {home_team} vs {away_team}...")

    # Récupérer les stats des équipes

    home_stats = compute_weighted_team_stats(team_stats, home_team)
    away_stats = compute_weighted_team_stats(team_stats, away_team)

    if home_stats is None or away_stats is None:
        print("❌ Erreur : Données manquantes pour une des équipes.")
        return None
    
# Créer les features du match
    match_features = {f"{key}_home": home_stats[key] for key in home_stats.keys()}
    match_features.update({f"{key}_away": away_stats[key] for key in away_stats.keys()})
# Vérifier que toutes les valeurs sont bien des nombres
    match_features = {key: float(value) for key, value in match_features.items()}

# Calcul du taux de victoires à domicile et défaites en extérieur
    home_win_rate = df[df["home_team"] == home_team]["winner"].mean()
    away_loss_rate = 1 - df[df["away_team"] == away_team]["winner"].mean()
# Si les taux de victoires/défaites sont NaN, mettre une valeur moyenne
    if pd.isna(home_win_rate):
        home_win_rate = df["home_win_rate"].mean()
    if pd.isna(away_loss_rate):
        away_loss_rate = df["away_loss_rate"].mean()
# Ajouter ces valeurs aux features du match
    match_features["home_win_rate"] = home_win_rate
    match_features["away_loss_rate"] = away_loss_rate


    # Créer les features du match
    match_features = {f"{key}_home": home_stats[key] for key in home_stats.keys()}
    match_features.update({f"{key}_away": away_stats[key] for key in away_stats.keys()})

# Vérifier si les taux existent dans les features du match, sinon les ajouter avec des valeurs moyennes
    if "home_win_rate" not in match_features:
        match_features["home_win_rate"] = df["home_win_rate"].mean()  # Valeur moyenne du dataset
    if "away_loss_rate" not in match_features:
        match_features["away_loss_rate"] = df["away_loss_rate"].mean()  # Valeur moyenne du dataset
    # Vérifier que les features correspondent à celles du modèle
    match_features_df = pd.DataFrame([match_features])[features]  # ⚠️ On garde les mêmes colonnes

    # Prédiction
    prediction_proba = rf_classifier.predict_proba(match_features_df)
    prob_home_win = prediction_proba[0][1]
    # Vérifier si le modèle prédit correctement un match connu
    test_game = df[df["season"] == 2024].sample(1)  # Prend un match au hasard en 2024
    test_features = test_game[features]
    real_winner = test_game["winner"].values[0]

# Faire la prédiction
    predicted_proba = rf_classifier.predict_proba(test_features)
    predicted_winner = 1 if predicted_proba[0][1] > 0.5 else 0

    print(f"📊 Match testé : {test_game[['home_team', 'away_team']]}")
    print(f"✅ Vainqueur réel : {real_winner}, 🏈 Prédiction : {predicted_winner}")



    print(f"📊 {home_team} a {prob_home_win * 100:.2f}% de chances de gagner.")
    print(f"📊 Statistiques utilisées pour la prédiction : {features}")

    return prob_home_win

# ========== 🏈 INTERFACE STREAMLIT ==========

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🏈 Prédiction de match NFL 🔮</h1>", unsafe_allow_html=True)

# Ajout d'un cadre pour la précision du modèle
st.sidebar.markdown("## 📊 Précision du Modèle")
st.sidebar.metric(label="Précision sur le test", value=f"{accuracy * 100:.2f}%")

st.sidebar.markdown("### 🎯 Informations supplémentaires")
st.sidebar.markdown("- Modèle : **RandomForestClassifier**")
st.sidebar.markdown("- Nombre d’arbres : **100**")
st.sidebar.markdown("- Profondeur max : **3**")

# Interface de sélection des équipes avec colonnes
col1, col2 = st.columns(2)

with col1:
    team1_input = st.text_input("🏠 Entrez l'équipe à domicile :", "")
    
with col2:
    team2_input = st.text_input("✈️ Entrez l'équipe à l'extérieur :", "")

# Bouton de prédiction stylisé
if st.button("🔮 Prédire le match", help="Cliquez pour obtenir une prédiction !"):
    team1 = normalize_team_name(team1_input.strip())
    team2 = normalize_team_name(team2_input.strip())

    available_teams = df["home_team"].unique()

    if team1 not in available_teams or team2 not in available_teams:
        st.error("❌ L'une des équipes n'est pas valide. Veuillez réessayer.")
    else:
        prob_home_win = predict_future_match(team1, team2)

        if prob_home_win is None:
            st.error("❌ Données manquantes pour ce match.")
        else:
            team1_full_name = TEAM_FULL_NAMES.get(team1, team1)
            team2_full_name = TEAM_FULL_NAMES.get(team2, team2)

            col3, col4 = st.columns(2)

            with col3:
                st.markdown(f"<h3 style='text-align: center; color: #1E90FF;'>{team1_full_name}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>{prob_home_win * 100:.2f}% de chances de gagner</h4>", unsafe_allow_html=True)
                st.progress(prob_home_win)

            with col4:
                st.markdown(f"<h3 style='text-align: center; color: #FF4500;'>{team2_full_name}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>{(1 - prob_home_win) * 100:.2f}% de chances de gagner</h4>", unsafe_allow_html=True)
                st.progress(1 - prob_home_win)

            # Affichage du vainqueur prédictif
            if prob_home_win > 0.5:
                st.success(f"🏆 **{team1_full_name} devrait gagner !**")
            else:
                st.success(f"🏆 **{team2_full_name} devrait gagner !**")

            # Graphique circulaire avec matplotlib
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.pie(
                [prob_home_win, 1 - prob_home_win],
                labels=[team1_full_name, team2_full_name],
                autopct='%1.1f%%',
                colors=["#1E90FF", "#FF4500"],
                startangle=90,
                wedgeprops={"edgecolor": "black"}
            )
            ax.set_title("📊 Probabilités de victoire")

            st.pyplot(fig)

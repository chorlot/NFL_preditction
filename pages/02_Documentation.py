# pages/02_Documentation.py
import streamlit as st
import pandas as pd
import nfl_data_py as nfl

st.set_page_config(page_title="Documentation • Acronymes NFL", page_icon="📘", layout="wide")

st.title("📘 Documentation — Acronymes des équipes NFL")

@st.cache_data
def load_team_acronyms() -> pd.DataFrame:
    # Récupère l'info officielle des équipes (abbr, nom, logo, conf, division, etc.)
    # Source: nfl_data_py.import_team_desc()
    desc = nfl.import_team_desc()  # renvoie un DataFrame avec 'team_abbr', 'team_name', etc.
    # On garde les colonnes utiles si elles existent :
    wanted = [c for c in ["team_abbr", "team_name", "team_nick", "team_city", "team_conf", "team_division"] if c in desc.columns]
    df = desc[wanted].drop_duplicates().sort_values(by=wanted[0] if wanted else None).reset_index(drop=True)

    # Construire un "Nom complet" parlant si on a les composantes
    if {"team_city", "team_nick"}.issubset(df.columns):
        df["full_name"] = (df["team_city"].fillna("") + " " + df["team_nick"].fillna("")).str.strip()
    elif {"team_name"}.issubset(df.columns):
        df["full_name"] = df["team_name"]
    else:
        df["full_name"] = df.get("team_abbr", pd.Series(dtype=str))

    # Mettre la colonne abréviation en premier
    cols = ["team_abbr", "full_name"] + [c for c in df.columns if c not in ("team_abbr", "full_name")]
    return df[cols]

df = load_team_acronyms()

st.markdown(
    "Cette page liste les **acronymes officiels** des franchises NFL directement depuis `nfl_data_py`."
)

# 🔎 Recherche
q = st.text_input("Rechercher (ville, surnom, abréviation…)", "")
if q:
    q_low = q.lower()
    df_view = df[df.apply(lambda r: any(q_low in str(v).lower() for v in r.values), axis=1)]
else:
    df_view = df

# 📥 Téléchargement
csv = df_view.to_csv(index=False).encode("utf-8")
st.download_button("Télécharger en CSV", csv, file_name="nfl_team_acronyms.csv")

# 📋 Tableau interactif
st.dataframe(
    df_view,
    use_container_width=True,
    hide_index=True
)

# 🧩 Petite légende
st.caption("Données d’équipes chargées avec `nfl.import_team_desc()` (acronymes, villes, surnoms, conférences, divisions).")

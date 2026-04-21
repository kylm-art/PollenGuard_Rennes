# app/main.py
# Lancement : uv run streamlit run app/main.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import joblib  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from datetime import date, datetime  # noqa: E402
from utils import (  # noqa: E402
    predire_risque_complet,
    predire_risque_futur,
    predire_depuis_s3,
    charger_s3_modele,
    charger_historique_pollen,
    charger_meteo_recente,
    charger_donnees_pour_date,   
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PollenGuard — Rennes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# CSS — THÈME PRINTEMPS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Police et fond général */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #F7FBF7;
}

/* Fond principal */
.stApp { background-color: #F7FBF7; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #D8F3DC !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stRadio label { color: #D8F3DC !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 500;
    font-size: 14px;
    color: #555;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2D6A4F, #40916C) !important;
    color: white !important;
}

/* Métriques */
[data-testid="stMetric"] {
    background: white;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #E8F5E9;
}
[data-testid="stMetricValue"] { font-size: 22px !important; }

/* Masquer éléments Streamlit */
#MainMenu, footer, header { visibility: hidden; }

/* Fix date input et radio dans sidebar */
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background-color: rgba(255,255,255,0.15) !important;
    color: #D8F3DC !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder {
    color: #95D5B2 !important;
}
[data-testid="stSidebar"] .stRadio > div {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 8px;
}
[data-testid="stSidebar"] .stRadio label {
    color: #D8F3DC !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #95D5B2 !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═════════════════════════════════════════════════════════════════════════════
RISK_STYLE = {
    "Faible":   {"color": "#2D6A4F", "bg": "#D8F3DC", "light": "#EAFAF1", "icon": "✅"},
    "À risque": {"color": "#C0392B", "bg": "#FADBD8", "light": "#FDEDEC", "icon": "🚨"},
}

MESSAGES = {
    ("Faible", "Faible"):   ("#D8F3DC", "#2D6A4F",
        "Bonne nouvelle ! Les deux types de pollen sont à des niveaux bas. "
        "Profitez de l'extérieur en toute tranquillité."),
    ("Faible", "À risque"): ("#FEF9E7", "#D4AC0D",
        "Prudence avec les graminées. Évitez les pelouses entre 12h–17h "
        "et pensez à votre traitement si vous êtes allergique."),
    ("À risque", "Faible"): ("#FEF9E7", "#D4AC0D",
        "Le bouleau est actif aujourd'hui. Gardez les fenêtres fermées, "
        "prenez votre traitement le matin et préférez sortir tôt ou en soirée."),
    ("À risque", "À risque"): ("#FADBD8", "#C0392B",
        "Journée difficile pour les allergiques ! Prenez votre traitement dès le matin, "
        "gardez les fenêtres fermées et limitez les sorties entre 10h et 18h."),
}

SENSIBILITE_SEUILS = {
    "Peu sensible 😊":       0.70,
    "Sensibilité normale 😐": 0.50,
    "Très sensible 😰":       0.30,
}

JOURS_FR = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
MOIS_FR  = ["","Janvier","Février","Mars","Avril","Mai","Juin",
            "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]


def fmt_date(ts):
    ts = pd.Timestamp(ts)
    return f"{JOURS_FR[ts.weekday()]} {ts.day} {MOIS_FR[ts.month]} {ts.year}"


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENTS CACHÉS
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def charger_modeles():
    root = Path(__file__).parent.parent / "models"
    m_b  = joblib.load(root / "modele_bouleau.pkl")
    m_g  = joblib.load(root / "modele_graminees.pkl")
    feat = joblib.load(root / "features.pkl")
    return m_b, m_g, feat


@st.cache_data(ttl=3600, show_spinner=False)
def charger_pour_date(date_cible: str):
    """
    Charge les données des 120 jours précédant date_cible.
    120 jours couvre depuis le 1er janvier pour le GDD annuel.
    Mis en cache par date — ne recharge que si la date change.
    """
    return charger_donnees_pour_date(date_cible, jours=120)

@st.cache_data(show_spinner=False)
def charger_s3():
    return charger_s3_modele()


@st.cache_data(ttl=3600, show_spinner=False)
def charger_historique():
    return charger_historique_pollen()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px;">
        <span style="font-size:48px;">🌿</span>
        <h2 style="margin:8px 0 0;font-size:20px;font-weight:700;
                   color:#D8F3DC;">PollenGuard</h2>
        <p style="margin:4px 0 0;font-size:12px;color:#95D5B2;">
            Rennes · Intelligence Allergie</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    date_pred = st.date_input(
        "📅 Date (onglet Prédictions)",
        value=date.today(),
        min_value=date(2021, 1, 10),
        max_value=date.today(),
        help="Pour les prévisions futures (J+1, J+2, J+3), voir l'onglet Prédictions"
    )

    st.divider()

    sensibilite = st.radio(
        "🎯 Mon profil de sensibilité",
        options=list(SENSIBILITE_SEUILS.keys()),
        index=1,
        help=(
            "Peu sensible : alerte uniquement si le modèle est très certain\n"
            "Normale : seuil standard (50%)\n"
            "Très sensible : alerte dès le moindre doute (30%)"
        ),
    )
    seuil = SENSIBILITE_SEUILS[sensibilite]


    st.divider()
    st.markdown("""
    <div style="font-size:11px;color:#95D5B2;line-height:1.9;">
    📡 Météo : Open-Meteo<br>
    🌿 Pollen : Air Quality API<br>
    🤖 RF Bouleau · LogReg Graminées<br>
    🗓️ Entraîné 2021–2024<br>
    ✅ Testé 2025–avril 2026<br>
    📍 Rennes (48.11°N, 1.67°W)
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENTS
# ═════════════════════════════════════════════════════════════════════════════
modele_b, modele_g, features = charger_modeles()

d        = pd.Timestamp(date_pred)
date_str = fmt_date(d)

with st.spinner(f"Chargement du bulletin du {date_str}..."):
    try:
        df_recent = charger_pour_date(str(date_pred))
    except Exception:
        df_recent = pd.DataFrame()
        st.warning("Données météo indisponibles pour cette date.")
    df_hist = charger_historique()

# ═════════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════
est_aujourdhui = (date_pred == date.today())
label_date_kpi = "aujourd'hui" if est_aujourdhui else date_str

st.markdown(f"""
<div style="background:white;border-radius:14px;padding:16px 24px;
            margin-bottom:20px;border-left:5px solid #2D6A4F;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div>
            <h3 style="margin:0;color:#1B4332;font-size:18px;">
                📊 Bulletin du {label_date_kpi}</h3>
            <p style="margin:4px 0 0;font-size:13px;color:#666;">
                Indicateurs calculés à partir des données de ce jour</p>
        </div>
        <div style="background:{'#D8F3DC' if est_aujourdhui else '#EBF5FB'};
                    border-radius:8px;padding:6px 14px;">
            <p style="margin:0;font-size:12px;font-weight:600;
                       color:{'#1B4332' if est_aujourdhui else '#1A5276'};">
                {'🔴 LIVE' if est_aujourdhui else '📅 ' + date_pred.strftime('%d/%m/%Y')}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🏠 Tableau de bord",
    "🔮 Prédictions",
    "📖 Documentation",
])


# ═══════════════════════════════════════════════════════════════════════════
#   ONGLET 1 — TABLEAU DE BORD (KPIs + graphiques)                         
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Calcul des KPIs ──────────────────────────────────────────────────────
    # Par — on cherche la ligne exacte de la date choisie
    if not df_recent.empty:
        mask_today = df_recent["date"].dt.date == date_pred
        today_row  = df_recent[mask_today].iloc[-1] \
            if mask_today.any() else df_recent.iloc[-1]
        idx        = df_recent[df_recent["date"].dt.date <= date_pred].index
        hier_row   = df_recent.iloc[idx[-2]] if len(idx) >= 2 else None
    else:
        today_row = hier_row = None

    # Et pour le GDD — prend l'année de la date choisie, pas l'année courante
    annee_courante = d.year
    df_jan = df_recent[
        (df_recent["date"].dt.year == annee_courante) &
        (df_recent["date"].dt.date <= date_pred)
    ].copy()

    # -- KPI 1 : Taxon Leader
    # Comparaison des concentrations de pollen_bouleau et pollen_graminees
    # du jour le plus récent disponible dans df_recent (données API dernières 24h)
    if today_row is not None:
        pb_today = float(today_row.get("pollen_bouleau", 0) or 0)
        pg_today = float(today_row.get("pollen_graminees", 0) or 0)
        if pb_today > pg_today:
            taxon_leader = "🌳 Bouleau"
            taxon_color  = "#2D6A4F"
            taxon_val    = f"{pb_today:.1f} gr/m³"
        elif pg_today > pb_today:
            taxon_leader = "🌾 Graminées"
            taxon_color  = "#D4AC0D"
            taxon_val    = f"{pg_today:.1f} gr/m³"
        else:
            taxon_leader = "Égalité"
            taxon_color  = "#555"
            taxon_val    = f"{pb_today:.1f} gr/m³"
    else:
        taxon_leader, taxon_color, taxon_val = "–", "#555", "–"

    # -- KPI 2 : Indice de dispersion par le vent
    # vitesse_vent du jour rapportée sur 10 (max référence = 50 km/h)
    # Un indice élevé = pollen dispersé loin → risque plus étendu géographiquement
    vent_today = float(today_row.get("vitesse_vent", 0) or 0) if today_row is not None else 0
    indice_vent = min(round(vent_today / 50 * 10, 1), 10.0)
    if indice_vent < 4:
        vent_msg = "Air calme — pollen concentré localement"
    elif indice_vent < 7:
        vent_msg = "Vent modéré — pollen dispersé sur plusieurs km"
    else:
        vent_msg = "Vent fort — pollen transporté très loin"

    # -- KPI 3 : Score de lessivage (pluie)
    # Cumul de precipitations des 3 derniers jours / seuil 15mm = 100% nettoyé
    # Plus le score est élevé, plus l'air a été "lavé" par la pluie récente
    precip_3j = df_recent["precipitations"].tail(3).sum() if not df_recent.empty else 0
    score_lessivage = min(int(precip_3j / 15 * 100), 100)
    if score_lessivage < 20:
        lessivage_msg = "Air sec — pollen non lavé"
    elif score_lessivage < 60:
        lessivage_msg = "Lessivage partiel de l'atmosphère"
    else:
        lessivage_msg = "Pluie récente — air bien nettoyé ✨"

    # -- KPI 4 : GDD cumulé depuis le 1er janvier de l'année en cours
    # Somme des (temperature - 5°C).clip(0) depuis le 1er janvier
    # Seuil de floraison du bouleau : ~300°C·jours cumulés
    annee_courante = d.year
    df_jan = df_recent[df_recent["date"].dt.year == annee_courante].copy()
    df_jan["gdd"] = (df_jan["temperature"] - 5).clip(lower=0)
    gdd_cumul = int(df_jan["gdd"].sum())
    seuil_bouleau = 300
    if gdd_cumul >= seuil_bouleau:
        gdd_msg = f"Seuil bouleau ({seuil_bouleau}°C·j) atteint — floraison active"
    else:
        gdd_msg = f"{seuil_bouleau - gdd_cumul}°C·j avant le seuil de floraison"

    # -- KPI 5 : Progression de la saison
    # Cumul pollen bouleau + graminées de l'année en cours
    # divisé par le cumul moyen annuel historique (2021-2024) — en pourcentage
    cumul_actuel = df_jan["pollen_bouleau"].sum() + df_jan["pollen_graminees"].sum() \
        if not df_jan.empty else 0
    if not df_hist.empty:
        cumul_hist_moy = (
            df_hist.groupby("annee").apply(
                lambda x: x["pollen_bouleau"].sum() + x["pollen_graminees"].sum()
            ).mean()
        )
        progression = min(int(cumul_actuel / cumul_hist_moy * 100), 100) \
            if cumul_hist_moy > 0 else 0
    else:
        progression = 0

    # -- KPI 6 : Anomalie saisonnière
    # (concentration_d - moyenne_historique_même_semaine) / moyenne * 100
    # Utilise la fenêtre ±7 jours autour du jour de l'année pour la comparaison
    if not df_hist.empty and today_row is not None:
        jour_courant = d.dayofyear
        df_fen_hist  = df_hist[
            (df_hist["jour_de_annee"] >= jour_courant - 7) &
            (df_hist["jour_de_annee"] <= jour_courant + 7)
        ]
        moy_hist_b = df_fen_hist["pollen_bouleau"].mean()
        moy_hist_g = df_fen_hist["pollen_graminees"].mean()
        if moy_hist_b > 0:
            anomalie_b = (pb_today - moy_hist_b) / moy_hist_b * 100
        else:
            anomalie_b = 0
        if moy_hist_g > 0:
            anomalie_g = (pg_today - moy_hist_g) / moy_hist_g * 100
        else:
            anomalie_g = 0
        anomalie = (anomalie_b + anomalie_g) / 2
        if abs(anomalie) < 15:
            anomalie_msg = "Dans la normale saisonnière"
            anomalie_color = "#2D6A4F"
        elif anomalie > 0:
            anomalie_msg = f"Saison {anomalie:+.0f}% au-dessus de la normale"
            anomalie_color = "#C0392B"
        else:
            anomalie_msg = f"Saison {anomalie:+.0f}% en dessous de la normale"
            anomalie_color = "#2980B9"
    else:
        anomalie, anomalie_msg, anomalie_color = 0, "Données insuffisantes", "#555"

    # -- KPI 7 : Décalage phénologique
    # Jour de début de saison = premier jour où pollen > 10 grains/m³
    # Comparaison entre 2026 (année courante dans df_jan) et moyenne 2021-2024
    if not df_hist.empty and not df_jan.empty:
        def debut_saison_bouleau(sub):
            t = sub[sub["pollen_bouleau"] > 10]
            return int(t["jour_de_annee"].min()) if not t.empty else np.nan \
                if "jour_de_annee" in sub.columns else np.nan

        def debut_saison_df_recent(df):
            df = df.copy()
            df["jour_de_annee"] = df["date"].dt.dayofyear
            t = df[df["pollen_bouleau"] > 10]
            return int(t["jour_de_annee"].min()) if not t.empty else np.nan

        debut_actuel = debut_saison_df_recent(df_jan)
        debut_hist   = df_hist.groupby("annee").apply(debut_saison_bouleau).mean()
        if not np.isnan(debut_actuel) and not np.isnan(debut_hist):
            decalage = int(debut_hist - debut_actuel)
            if decalage > 3:
                decalage_msg   = f"{abs(decalage)} jours d'avance sur la moyenne"
                decalage_color = "#E67E22"
            elif decalage < -3:
                decalage_msg   = f"{abs(decalage)} jours de retard sur la moyenne"
                decalage_color = "#2980B9"
            else:
                decalage_msg   = "Démarrage dans les temps habituels"
                decalage_color = "#2D6A4F"
        else:
            decalage, decalage_msg, decalage_color = 0, "Données insuffisantes", "#555"
    else:
        decalage, decalage_msg, decalage_color = 0, "Données insuffisantes", "#555"

    # ── Titre section KPIs ────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:white;border-radius:14px;padding:16px 24px;
                margin-bottom:20px;border-left:5px solid #2D6A4F;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <h3 style="margin:0;color:#1B4332;font-size:18px;">
            📊 Indicateurs du jour</h3>
        <p style="margin:4px 0 0;font-size:13px;color:#666;">
            Basés sur les données météo et pollen des dernières 24–72h</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Ligne 1 de KPIs ───────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {taxon_color};height:120px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">🏆 Taxon dominant</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{taxon_color};">{taxon_leader}</p>
            <p style="margin:0;font-size:12px;color:#666;">{taxon_val}</p>
        </div>""", unsafe_allow_html=True)

    with k2:
        vent_color = "#2D6A4F" if indice_vent < 4 else "#E67E22" if indice_vent < 7 else "#C0392B"
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {vent_color};height:120px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">💨 Indice vent</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{vent_color};">{indice_vent}/10</p>
            <p style="margin:0;font-size:12px;color:#666;">{vent_msg}</p>
        </div>""", unsafe_allow_html=True)

    with k3:
        less_color = "#C0392B" if score_lessivage < 20 else "#E67E22" if score_lessivage < 60 else "#2D6A4F"
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {less_color};height:120px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">🌧️ Score lessivage</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{less_color};">{score_lessivage}%</p>
            <p style="margin:0;font-size:12px;color:#666;">{lessivage_msg}</p>
        </div>""", unsafe_allow_html=True)

    with k4:
        gdd_color = "#2D6A4F" if gdd_cumul >= seuil_bouleau else "#E67E22"
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {gdd_color};height:120px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">🌡️ Chaleur cumulée</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{gdd_color};">{gdd_cumul} °C·j</p>
            <p style="margin:0;font-size:12px;color:#666;">{gdd_msg}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ligne 2 de KPIs ───────────────────────────────────────────────────────
    k5, k6, k7, k8 = st.columns(4)

    with k5:
        # Progression de la saison — barre de progression visuelle
        prog_color = "#2D6A4F" if progression < 60 else "#E67E22" if progression < 85 else "#C0392B"
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {prog_color};height:140px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">📈 Progression saison</p>
            <p style="margin:8px 0 6px;font-size:22px;font-weight:700;
                       color:{prog_color};">{progression}%</p>
            <div style="background:#eee;border-radius:6px;height:8px;">
                <div style="background:{prog_color};width:{progression}%;
                            height:8px;border-radius:6px;"></div>
            </div>
            <p style="margin:6px 0 0;font-size:12px;color:#666;">
                {"Le plus dur est passé 🎉" if progression > 65 else "Saison en cours"}</p>
        </div>""", unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {anomalie_color};height:140px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">📊 Anomalie saisonnière</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{anomalie_color};">{anomalie:+.0f}%</p>
            <p style="margin:0;font-size:12px;color:#666;">{anomalie_msg}</p>
        </div>""", unsafe_allow_html=True)

    with k7:
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {decalage_color};height:140px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">🗓️ Décalage phénologique</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{decalage_color};">
                {"+" if decalage > 0 else ""}{decalage} j</p>
            <p style="margin:0;font-size:12px;color:#666;">{decalage_msg}</p>
        </div>""", unsafe_allow_html=True)

    with k8:
        # Pic max observé cette année vs historique
        pic_actuel = max(df_jan["pollen_bouleau"].max(), df_jan["pollen_graminees"].max()) \
            if not df_jan.empty else 0
        pic_hist_moy = max(
            df_hist.groupby("annee")["pollen_bouleau"].max().mean(),
            df_hist.groupby("annee")["pollen_graminees"].max().mean()
        ) if not df_hist.empty else 1
        delta_pic = (pic_actuel / pic_hist_moy - 1) * 100 if pic_hist_moy > 0 else 0
        pic_color = "#C0392B" if delta_pic > 20 else "#2D6A4F" if delta_pic < -20 else "#E67E22"
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
                    border-top:4px solid {pic_color};height:140px;">
            <p style="margin:0;font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:1px;">🎯 Pic max {annee_courante}</p>
            <p style="margin:8px 0 4px;font-size:22px;font-weight:700;
                       color:{pic_color};">{pic_actuel:.0f} gr/m³</p>
            <p style="margin:0;font-size:12px;color:#666;">
                {delta_pic:+.0f}% vs moyenne historique</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphique — Évolution pollen récente ──────────────────────────────────
    st.markdown("""
    <div style="background:white;border-radius:14px;padding:16px 24px;
                margin-bottom:12px;border-left:5px solid #52B788;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <h3 style="margin:0;color:#1B4332;font-size:16px;">
            📉 Évolution du pollen — 30 derniers jours</h3>
    </div>
    """, unsafe_allow_html=True)

    if not df_recent.empty:
        df_plot = df_recent.tail(30).copy()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["pollen_bouleau"],
            name="🌳 Bouleau",
            line=dict(color="#2D6A4F", width=2.5),
            fill="tozeroy", fillcolor="rgba(45,106,79,0.1)",
            hovertemplate="%{x|%d/%m}<br>Bouleau : %{y:.1f} gr/m³<extra></extra>",
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_plot["date"], y=df_plot["pollen_graminees"],
            name="🌾 Graminées",
            line=dict(color="#D4AC0D", width=2.5),
            fill="tozeroy", fillcolor="rgba(212,172,13,0.1)",
            hovertemplate="%{x|%d/%m}<br>Graminées : %{y:.1f} gr/m³<extra></extra>",
        ))
        fig_trend.add_hline(y=30, line_dash="dot", line_color="#C0392B",
                             line_width=1.5,
                             annotation_text="Seuil risque (30 gr/m³)",
                             annotation_font_color="#C0392B",
                             annotation_position="top left")
        fig_trend.update_layout(
            height=280, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20,r=20,t=20,b=20),
            xaxis=dict(showgrid=True, gridcolor="#F0F0F0", title=""),
            yaxis=dict(showgrid=True, gridcolor="#F0F0F0",
                       title="Concentration (gr/m³)"),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            font=dict(family="Inter, sans-serif", size=12),
        )
        st.plotly_chart(fig_trend, width="stretch")

    # ── Graphique — Comparaison historique ───────────────────────────────────
    if not df_hist.empty:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:16px 24px;
                    margin:12px 0;border-left:5px solid #52B788;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);">
            <h3 style="margin:0;color:#1B4332;font-size:16px;">
                📊 Comparaison avec les années précédentes (±30 jours)</h3>
        </div>
        """, unsafe_allow_html=True)

        jour_cible = d.dayofyear
        df_fen = df_hist[
            (df_hist["jour_de_annee"] >= d.dayofyear - 30) &
            (df_hist["jour_de_annee"] <= d.dayofyear + 30) &
            (df_hist["annee"] != d.year)   # ← exclut l'année sélectionnée du "historique"
            ]

        tab_b, tab_g = st.tabs(["🌳 Bouleau", "🌾 Graminées"])
        for tab, col_p, nom, coul in [
            (tab_b, "pollen_bouleau",   "Bouleau",   "#2D6A4F"),
            (tab_g, "pollen_graminees", "Graminées", "#D4AC0D"),
        ]:
            with tab:
                fig_hist = go.Figure()
                for annee in sorted(df_fen["annee"].unique()):
                    sub = df_fen[df_fen["annee"] == annee].sort_values("jour_de_annee")
                    fig_hist.add_trace(go.Scatter(
                        x=sub["jour_de_annee"], y=sub[col_p],
                        name=str(annee),
                        line=dict(width=2, color=coul if annee == max(df_fen["annee"]) else "#CCCCCC",
                                  dash="solid" if annee == max(df_fen["annee"]) else "dot"),
                        opacity=1.0 if annee == max(df_fen["annee"]) else 0.5,
                        hovertemplate=f"{annee} — Jour %{{x}}<br>{nom} : %{{y:.1f}} gr/m³<extra></extra>",
                    ))
                fig_hist.add_vline(x=jour_cible, line_dash="dash",
                                    line_color="#333", line_width=1.5)
                fig_hist.add_hline(y=30, line_dash="dot", line_color="#C0392B",
                                    line_width=1,
                                    annotation_text="Seuil risque",
                                    annotation_font_color="#C0392B",
                                    annotation_position="top left")
                fig_hist.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20,r=20,t=20,b=20),
                    xaxis=dict(title="Jour de l'année",
                               tickvals=[1,60,121,182,244,305],
                               ticktext=["Janv.","Mars","Mai","Juil.","Sept.","Nov."],
                               showgrid=True, gridcolor="#F0F0F0"),
                    yaxis=dict(title="Concentration (gr/m³)",
                               showgrid=True, gridcolor="#F0F0F0"),
                    legend=dict(orientation="h", y=1.08),
                    hovermode="x unified",
                    font=dict(family="Inter, sans-serif", size=12),
                )
                st.plotly_chart(fig_hist, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════
#   ONGLET 2 — PRÉDICTIONS                                                  
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Prédiction pour la date choisie ──────────────────────────────────────
    d        = pd.Timestamp(date_pred)
    date_str = fmt_date(d)

    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:16px 24px;
                margin-bottom:20px;border-left:5px solid #2D6A4F;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <h3 style="margin:0;color:#1B4332;font-size:18px;">
            🔍 Prédiction pour le {date_str}</h3>
        <p style="margin:4px 0 0;font-size:13px;color:#666;">
            Profil appliqué : <b>{sensibilite}</b>
            (seuil de décision : {int(seuil*100)}%)</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Analyse en cours..."):
        try:
            resultat = predire_risque_complet(
                str(date_pred), modele_b, modele_g, features, seuil=seuil
            )
        except Exception:
            df_s3 = charger_s3()
            if df_s3.empty:
                st.error("API et cache S3 indisponibles. Réessayez plus tard.")
                st.stop()
            try:
                resultat = predire_depuis_s3(
                    str(date_pred), modele_b, modele_g, features, df_s3
                )
            except ValueError:
                st.error(
                    f"Données indisponibles pour le {date_str}. "
                    "API inaccessible et date hors cache S3 (disponible jusqu'au 19/04/2026)."
                )
                st.stop()

    if resultat.get("source") == "cache":
        st.info("ℹ️ Prédiction depuis le cache S3 (API indisponible).")

    rb = RISK_STYLE[resultat["bouleau"]]
    rg = RISK_STYLE[resultat["graminees"]]

    # Cartes résultat
    col1, col2 = st.columns(2)
    for col, emoji, nom, label, r, proba_key in [
        (col1, "🌳", "Bouleau",   resultat["bouleau"],   rb, "proba_bouleau"),
        (col2, "🌾", "Graminées", resultat["graminees"], rg, "proba_graminees"),
    ]:
        proba = resultat.get(proba_key)
        proba_txt = f"{proba*100:.0f}% de probabilité de risque" if proba is not None else ""
        with col:
            st.markdown(f"""
            <div style="background:{r['bg']};border-radius:14px;padding:28px;
                        text-align:center;border:2px solid {r['color']};
                        box-shadow:0 4px 16px rgba(0,0,0,0.08);">
                <p style="margin:0;font-size:18px;color:#444;font-weight:500;">
                    {emoji} Pollen de {nom}</p>
                <p style="margin:12px 0 8px;font-size:52px;line-height:1;">{r['icon']}</p>
                <p style="margin:0;font-size:26px;font-weight:700;color:{r['color']};">
                    {label}</p>
                <p style="margin:8px 0 0;font-size:13px;color:#666;font-style:italic;">
                    {proba_txt}</p>
            </div>
            """, unsafe_allow_html=True)

    # Niveau de confiance du modèle
    if resultat.get("proba_bouleau") is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        pb = resultat["proba_bouleau"]
        pg = resultat["proba_graminees"]
        confiance_b = max(pb, 1-pb) * 100
        confiance_g = max(pg, 1-pg) * 100
        st.markdown("""
        <div style="background:white;border-radius:12px;padding:16px 20px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:12px;">
            <p style="margin:0 0 10px;font-size:14px;font-weight:600;color:#333;">
                🎯 Niveau de confiance du modèle</p>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div>
                <div style="display:flex;justify-content:space-between;
                            font-size:13px;margin-bottom:4px;">
                    <span>🌳 Bouleau</span>
                    <span style="font-weight:600;">{confiance_b:.0f}%</span>
                </div>
                <div style="background:#eee;border-radius:6px;height:8px;">
                    <div style="background:#2D6A4F;width:{confiance_b:.0f}%;
                                height:8px;border-radius:6px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div>
                <div style="display:flex;justify-content:space-between;
                            font-size:13px;margin-bottom:4px;">
                    <span>🌾 Graminées</span>
                    <span style="font-weight:600;">{confiance_g:.0f}%</span>
                </div>
                <div style="background:#eee;border-radius:6px;height:8px;">
                    <div style="background:#D4AC0D;width:{confiance_g:.0f}%;
                                height:8px;border-radius:6px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Message interprétation
    st.markdown("<br>", unsafe_allow_html=True)
    bg, border, msg = MESSAGES[(resultat["bouleau"], resultat["graminees"])]
    st.markdown(f"""
    <div style="background:{bg};border-left:6px solid {border};
                border-radius:10px;padding:18px 22px;margin-bottom:20px;">
        <p style="margin:0;font-size:15px;color:#333;line-height:1.7;">
            💬 <b>Ce que ça veut dire pour vous :</b> {msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # Créneaux horaires
    risque_global = (
        resultat["bouleau"] == "À risque" or
        resultat["graminees"] == "À risque"
    )
    st.markdown("### 🕐 Créneaux recommandés")
    if not risque_global:
        st.success("✅ **Toute la journée** — Conditions favorables !")
    else:
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("""
            <div style="background:#D8F3DC;border-radius:10px;padding:18px;
                        text-align:center;border:2px solid #2D6A4F;">
                <p style="margin:0;font-size:13px;color:#1B4332;">Moments favorables</p>
                <p style="margin:8px 0 0;font-size:20px;font-weight:700;
                           color:#1B4332;">⏰ Avant 9h · Après 19h</p>
            </div>""", unsafe_allow_html=True)
        with h2:
            st.markdown("""
            <div style="background:#FADBD8;border-radius:10px;padding:18px;
                        text-align:center;border:2px solid #C0392B;">
                <p style="margin:0;font-size:13px;color:#922B21;">À éviter</p>
                <p style="margin:8px 0 0;font-size:20px;font-weight:700;
                           color:#922B21;">⛔ 10h – 18h</p>
            </div>""", unsafe_allow_html=True)

    # ── Prévisions J+1, J+2, J+3 ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:white;border-radius:14px;padding:16px 24px;
                margin-bottom:20px;border-left:5px solid #52B788;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <h3 style="margin:0;color:#1B4332;font-size:18px;">
            🌤️ Prévisions J+1, J+2, J+3</h3>
        <p style="margin:4px 0 0;font-size:12px;color:#666;">
            Basées sur les prévisions météo Open-Meteo — fiabilité décroissante
            (J+1 > J+2 > J+3)</p>
    </div>
    """, unsafe_allow_html=True)

    previsions = []
    for j in [1, 2, 3]:
        try:
            prev = predire_risque_futur(j, modele_b, modele_g, features, seuil=seuil)
            previsions.append(prev)
        except Exception:
            previsions.append(None)

    if any(p is not None for p in previsions):
        from datetime import timedelta
        p_cols = st.columns(3)
        labels_j = ["Demain", "Dans 2 jours", "Dans 3 jours"]
        fiabilite = ["⭐⭐⭐ Très fiable", "⭐⭐ Fiable", "⭐ Estimation"]

        for col, prev, lbl, fib in zip(p_cols, previsions, labels_j, fiabilite):
            with col:
                if prev is None:
                    st.markdown(f"""
                    <div style="background:#f8f8f8;border-radius:12px;padding:20px;
                                text-align:center;border:1px solid #eee;">
                        <p style="margin:0;font-size:12px;color:#888;">{lbl}</p>
                        <p style="margin:8px 0;font-size:20px;">⚠️</p>
                        <p style="margin:0;font-size:13px;color:#999;">Indisponible</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    rb_p = RISK_STYLE[prev["bouleau"]]
                    rg_p = RISK_STYLE[prev["graminees"]]
                    pire = "À risque" if "À risque" in [prev["bouleau"], prev["graminees"]] else "Faible"
                    r_p  = RISK_STYLE[pire]
                    d_p  = pd.Timestamp(prev["date"])
                    st.markdown(f"""
                    <div style="background:{r_p['light']};border-radius:12px;
                                padding:18px;text-align:center;
                                border:2px solid {r_p['color']};
                                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                        <p style="margin:0;font-size:11px;color:#888;">{lbl}</p>
                        <p style="margin:2px 0;font-size:12px;font-weight:600;
                                   color:{r_p['color']};">
                            {d_p.strftime('%A %d %b').capitalize()}</p>
                        <p style="margin:8px 0;font-size:32px;line-height:1;">
                            {r_p['icon']}</p>
                        <p style="margin:0 0 6px;font-size:11px;color:#888;">
                            🌳 {prev['bouleau']} &nbsp;|&nbsp; 🌾 {prev['graminees']}</p>
                        <p style="margin:0;font-size:10px;color:#aaa;">{fib}</p>
                    </div>""", unsafe_allow_html=True)
    else:
        st.info("Prévisions indisponibles pour le moment.")

    # ── Explication fonctionnement ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Comment fonctionnent ces prédictions ?"):
        st.markdown(f"""
        **Pour une date passée** : le modèle récupère les 20 derniers jours
        de météo et de pollen depuis l'API Open-Meteo Archive, reconstruit
        les features (lags, moyennes glissantes, GDD...) et prédit.

        **Pour J+1, J+2, J+3** : combinaison des archives récentes
        (pour les lags historiques) + prévisions météo Open-Meteo Forecast.
        Les prévisions à J+3 sont moins fiables car les erreurs météo s'accumulent.

        **Votre profil de sensibilité ({sensibilite})** :
        Le modèle estime une probabilité de risque entre 0 et 100%.
        Avec un seuil à **{int(seuil*100)}%**, une alerte se déclenche
        dès que le modèle pense qu'il y a {int(seuil*100)}% de chances
        que le pollen soit à risque.

        | Ce que le modèle analyse | Pourquoi |
        |---|---|
        | 🌡️ Températures récentes | Chaleur accumulée = floraison imminente |
        | 🌧️ Pluie récente | La pluie lave l'air |
        | 💨 Vent récent | Transporte le pollen sur des dizaines de km |
        | 🌿 Pollen des 3 derniers jours | La saison évolue progressivement |

        **Seuil de risque :** > 30 grains/m³ *(recommandation RNSA)*

        *Données : Open-Meteo Archive + Forecast + Air Quality API*
        *Modèles entraînés sur Rennes 2021–2024*
        """)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ONGLET 3 — DOCUMENTATION                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab3:

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1B4332, #2D6A4F);
                border-radius:14px;padding:28px 36px;margin-bottom:28px;
                box-shadow:0 4px 20px rgba(27,67,50,0.2);">
        <h2 style="margin:0;color:white;font-size:24px;font-weight:700;">
            📖 Documentation PollenGuard</h2>
        <p style="margin:8px 0 0;color:#B7E4C7;font-size:14px;">
            Comprendre les indicateurs, les modèles et les données utilisées</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1 : Comment ça marche ────────────────────────────────────────
    st.markdown("""
    <div style="background:white;border-radius:14px;padding:20px 28px;
                margin-bottom:16px;border-left:5px solid #2D6A4F;
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
        <h3 style="margin:0 0 16px;color:#1B4332;">🔬 Comment fonctionne l'application ?</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    PollenGuard utilise deux modèles de machine learning entraînés sur 4 ans de données
    (Rennes, 2021–2024) pour prédire le risque allergique journalier. Pour chaque
    prédiction, l'application récupère en temps réel les données météo et pollen
    des 20 derniers jours depuis l'API Open-Meteo, reconstruit les variables
    explicatives, et applique les modèles.

    L'application distingue deux cas :
    - **Dates passées** : données météo réelles depuis l'API archive Open-Meteo
    - **J+1 à J+3** : données météo prévisionnelles depuis l'API forecast Open-Meteo
      (fiabilité décroissante avec l'horizon)
    """)

    # ── Section 2 : Données sources ───────────────────────────────────────────
    with st.expander("📡 Sources de données", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div style="background:#F0FAF4;border-radius:10px;padding:16px;
                        border:1px solid #B7E4C7;">
            <h4 style="margin:0 0 10px;color:#1B4332;">🌡️ Données météo</h4>

            **Source :** Open-Meteo Archive API
            `archive-api.open-meteo.com`

            **Variables récupérées :**
            - Température horaire (°C) → agrégée en moyenne journalière
            - Précipitations horaires (mm) → agrégées en cumul journalier
            - Vitesse du vent horaire (km/h) → agrégée en moyenne journalière

            **Couverture :** 2021 – aujourd'hui
            **Localisation :** Rennes (48.11°N, 1.67°W)
            **Délai :** 1 à 2 jours (données archive)
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div style="background:#FEF9E7;border-radius:10px;padding:16px;
                        border:1px solid #F9E79F;">
            <h4 style="margin:0 0 10px;color:#7D6608;">🌿 Données pollen</h4>

            **Source :** Open-Meteo Air Quality API
            `air-quality-api.open-meteo.com`

            **Variables récupérées :**
            - `birch_pollen` → pollen de bouleau (grains/m³)
            - `grass_pollen` → pollen de graminées (grains/m³)

            Agrégées en **maximum journalier** (pic de concentration).

            **Couverture :** 2021 – aujourd'hui
            **Délai :** 1 à 2 jours (données archive)
            </div>
            """, unsafe_allow_html=True)

    # ── Section 3 : Variables explicatives ───────────────────────────────────
    with st.expander("⚙️ Variables explicatives (features) du modèle"):
        st.markdown("""
        Le modèle ne reçoit jamais les concentrations brutes du jour J — il utilise
        uniquement des informations du **passé** pour éviter toute fuite de données.
        """)

        st.markdown("""
        | Variable | Calcul | Justification |
        |---|---|---|
        | `jour_sin` / `jour_cos` | sin/cos(2π × jour_de_l_année / 365) | Encodage cyclique — le 31 déc. est proche du 1er jan. |
        | `mois_sin` / `mois_cos` | sin/cos(2π × mois / 12) | Saisonnalité mensuelle sans discontinuité |
        | `saison` | 0=hiver, 1=printemps, 2=été, 3=automne | Signal saisonnier grossier pour les modèles à arbres |
        | `annee` | Année entière (2021, 2022…) | Capture les tendances inter-annuelles |
        | `temp_lag1` | Température de J-1 | Effet immédiat : chaleur hier → pollen aujourd'hui |
        | `temp_roll7` | Moyenne température J-7 à J-1 | Accumulation thermique sur une semaine |
        | `gdd_cumul` | Σ(temperature - 5°C)⁺ depuis le 1er jan. | Proxy biologique du stade phénologique de la plante |
        | `precip_lag1` | Précipitations de J-1 | Lessivage atmosphérique immédiat |
        | `precip_lag2` | Précipitations de J-2 | Effet résiduel 48h après la pluie |
        | `precip_roll7` | Cumul précipitations J-7 à J-1 | Période sèche prolongée = accumulation de pollen |
        | `vitesse_vent_lag1` | Vent de J-1 | Dispersion locale la veille |
        | `vitesse_vent_roll7` | Moyenne vent J-7 à J-1 | Régime de vent de la semaine (site côtier) |
        | `pollen_X_lag1/2/3` | Pollen J-1, J-2, J-3 | La saison évolue progressivement — meilleur prédicteur |
        | `pollen_X_moy3j` | Moyenne pollen J-3 à J-1 | Lisse les pics ponctuels |
        """)

    # ── Section 4 : Les modèles ───────────────────────────────────────────────
    with st.expander("🤖 Les modèles de prédiction"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            <div style="background:#F0FAF4;border-radius:10px;padding:16px;
                        border:1px solid #B7E4C7;">
            <h4 style="margin:0 0 10px;color:#1B4332;">🌳 Modèle Bouleau</h4>

            **Algorithme :** Random Forest Classifier

            **Principe :** Ensemble de 300 arbres de décision entraînés
            sur des sous-échantillons aléatoires des données.
            Chaque arbre vote et la majorité l'emporte.

            **Avantages :**
            - Robuste aux valeurs aberrantes
            - Gère bien les interactions non-linéaires
            - Fournit des probabilités fiables

            **Variable cible :** Risque bouleau J+1
            (0 = Faible ≤30 gr/m³ · 1 = À risque >30 gr/m³)
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown("""
            <div style="background:#FEF9E7;border-radius:10px;padding:16px;
                        border:1px solid #F9E79F;">
            <h4 style="margin:0 0 10px;color:#7D6608;">🌾 Modèle Graminées</h4>

            **Algorithme :** Régression Logistique

            **Principe :** Modèle linéaire qui estime la probabilité
            d'appartenir à la classe "À risque" via une fonction
            sigmoïde appliquée à une combinaison linéaire des features.

            **Avantages :**
            - Très interprétable
            - Stable et rapide
            - Bien calibré en probabilités

            **Variable cible :** Risque graminées J+1
            (0 = Faible ≤30 gr/m³ · 1 = À risque >30 gr/m³)
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        **Entraînement :** 2021–2024 · **Test :** 2025 – avril 2026

        Le split est **temporel** (pas aléatoire) : on entraîne sur le passé
        et on teste sur le futur, comme le ferait un vrai système de prévision.
        Un split aléatoire laisserait "fuiter" des informations du futur dans
        l'entraînement et donnerait des scores artificiellement bons.
        """)

    # ── Section 5 : Les indicateurs ───────────────────────────────────────────
    with st.expander("📊 Comment sont calculés les indicateurs ?"):

        indicateurs = [
            (
                "🏆 Taxon dominant",
                "pollen_bouleau et pollen_graminees du jour sélectionné",
                "Comparaison directe des deux concentrations. "
                "L'espèce avec la valeur la plus haute domine la pollution pollinique du jour.",
                "Concentration bouleau = 45 gr/m³, graminées = 12 gr/m³ → Bouleau dominant"
            ),
            (
                "💨 Indice de dispersion par le vent",
                "vitesse_vent du jour sélectionné",
                "Formule : min(vitesse_vent / 50 × 10, 10). "
                "La valeur 50 km/h est la référence de vent fort en météorologie côtière. "
                "Un indice élevé signifie que le pollen local est transporté loin — "
                "même sans végétation à proximité, l'exposition reste possible.",
                "Vent = 25 km/h → 25/50 × 10 = 5/10 (dispersion modérée)"
            ),
            (
                "🌧️ Score de lessivage",
                "precipitations des 3 derniers jours (J-1, J-2, J-3)",
                "Formule : min(Σ précipitations 3j / 15mm × 100, 100%). "
                "Le seuil de 15mm correspond au cumul nécessaire pour une "
                "réduction significative des concentrations polliniques atmosphériques "
                "(source : études de la Société Française d'Allergologie).",
                "3mm + 5mm + 4mm = 12mm → 12/15 × 100 = 80% de lessivage"
            ),
            (
                "🌡️ Chaleur cumulée (GDD)",
                "temperature depuis le 1er janvier de l'année de la date choisie",
                "Formule : Σ max(température_journalière - 5°C, 0) depuis le 1er janvier. "
                "La base 5°C est le seuil standard en phénologie végétale "
                "en dessous duquel la croissance est nulle. "
                "Le seuil de 300°C·jours pour le bouleau est issu de la littérature "
                "phénologique européenne sur Betula pendula.",
                "Si le 15 avril, la somme des températures >5°C depuis le 1er jan. = 320°C·j → seuil atteint"
            ),
            (
                "📈 Progression de la saison",
                "pollen_bouleau + pollen_graminees depuis le 1er jan. (df_recent) "
                "vs cumul moyen annuel 2021–2025 (df_hist)",
                "Formule : Σ(pollen_b + pollen_g depuis 1er jan.) / moyenne(Σ annuelle historique) × 100. "
                "Indique quelle fraction du 'stock pollinique annuel moyen' a déjà été libérée. "
                "Au-delà de 65%, les concentrations maximales sont généralement passées.",
                "Cumul actuel = 1 200 · Cumul moyen historique = 2 000 → 60% de la saison écoulée"
            ),
            (
                "📊 Anomalie saisonnière",
                "pollen_bouleau + pollen_graminees du jour vs df_hist fenêtre ±7 jours",
                "Formule : (concentration_actuelle - moyenne_historique_±7j) / moyenne_historique × 100%. "
                "La fenêtre de ±7 jours permet de comparer la concentration du jour "
                "à ce qui se passait normalement à la même période des années précédentes, "
                "en lissant la variabilité journalière.",
                "Aujourd'hui = 80 gr/m³ · Moyenne historique ±7j = 55 gr/m³ → anomalie = +45%"
            ),
            (
                "🗓️ Décalage phénologique",
                "pollen_bouleau dans df_recent et df_hist",
                "Formule : jour_moyen_debut_historique - jour_debut_annee_courante. "
                "Le début de saison est défini comme le premier jour où pollen_bouleau > 10 gr/m³ "
                "(seuil RNSA de début de saison). "
                "Un résultat positif signifie que la saison a commencé plus tôt que la moyenne.",
                "Début moyen historique = jour 95 (5 avril) · Début 2026 = jour 83 (24 mars) → 12 jours d'avance"
            ),
            (
                "🎯 Pic maximum",
                "max(pollen_bouleau, pollen_graminees) depuis le 1er jan. de l'année choisie",
                "Maximum observé depuis le 1er janvier, comparé au maximum moyen annuel "
                "calculé sur df_hist. Indique si la saison est exceptionnellement intense "
                "ou au contraire clémente.",
                "Pic 2026 = 520 gr/m³ · Pic moyen historique = 380 gr/m³ → +37% vs normale"
            ),
        ]

        for titre, donnees, calcul, exemple in indicateurs:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:18px 20px;
                        margin-bottom:12px;border:1px solid #E8F5E9;
                        box-shadow:0 1px 4px rgba(0,0,0,0.04);">
                <h4 style="margin:0 0 10px;color:#1B4332;font-size:15px;">{titre}</h4>
                <table style="width:100%;font-size:13px;border-collapse:collapse;">
                    <tr>
                        <td style="padding:4px 8px 4px 0;color:#888;
                                   white-space:nowrap;vertical-align:top;
                                   width:120px;">📂 Données</td>
                        <td style="padding:4px 0;color:#333;">{donnees}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px 4px 0;color:#888;
                                   white-space:nowrap;vertical-align:top;">🔢 Calcul</td>
                        <td style="padding:4px 0;color:#333;">{calcul}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px 4px 0;color:#888;
                                   white-space:nowrap;vertical-align:top;">💡 Exemple</td>
                        <td style="padding:4px 0;font-style:italic;
                                   color:#2D6A4F;">{exemple}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 6 : Seuils et profils ─────────────────────────────────────────
    with st.expander("🎯 Seuils de risque et profils de sensibilité"):

        st.markdown("""
        **Seuil de classification :** 30 grains/m³

        Le seuil de 30 grains/m³ est la recommandation du **Réseau National de
        Surveillance Aérobiologique (RNSA)** comme frontière entre exposition
        faible et exposition à risque pour les personnes allergiques.
        """)

        st.markdown("""
        | Niveau | Concentration | Signification clinique |
        |---|---|---|
        | ✅ **Faible** | ≤ 30 gr/m³ | Peu de risque pour la majorité des allergiques |
        | 🚨 **À risque** | > 30 gr/m³ | Symptômes probables chez les personnes sensibles |
        """)

        st.divider()

        st.markdown("**Profils de sensibilité** — personnalisent le seuil de décision du modèle :")

        st.markdown("""
        | Profil | Seuil de probabilité | Effet |
        |---|---|---|
        | 😊 Peu sensible | 70% | Alerte uniquement si le modèle est très certain (>70% de probabilité de risque) |
        | 😐 Sensibilité normale | 50% | Seuil standard — alerte si le modèle pense que c'est plus risqué que pas risqué |
        | 😰 Très sensible | 30% | Alerte au moindre doute — recommandé pour les asthmatiques et allergiques sévères |

        Le modèle estime une **probabilité de risque** entre 0 et 100%.
        Le profil détermine à partir de quelle probabilité on affiche "À risque".
        """)

        st.info(
            "💡 **Conseil médical :** Les profils de sensibilité sont des outils d'aide "
            "à la décision. Ils ne remplacent pas l'avis de votre allergologue. "
            "Si vous souffrez d'asthme allergique, consultez votre médecin pour "
            "définir votre plan d'action personnalisé."
        )

    # ── Section 7 : Limites ────────────────────────────────────────────────────
    with st.expander("⚠️ Limites et précautions d'usage"):
        st.markdown("""
        **Limites techniques :**
        - Le modèle est entraîné sur Rennes uniquement — les résultats
          ne sont pas directement transposables à d'autres villes
        - Les prévisions à J+2 et J+3 dépendent des prévisions météo
          qui sont elles-mêmes imprécises à cet horizon
        - L'API Open-Meteo publie ses données avec 1 à 2 jours de délai —
          les données d'aujourd'hui ne sont disponibles que demain matin
        - Le pollen de l'API Air Quality est un modèle atmosphérique,
          pas une mesure terrain (contrairement au Réseau Sentinelles)

        **Limites médicales :**
        - PollenGuard est un outil d'information, pas un outil médical
        - La réaction allergique individuelle dépend aussi du terrain atopique,
          des co-expositions (pollution, tabac) et des traitements en cours
        - En cas de symptômes sévères, consultez un médecin

        **Reproductibilité :**
        - Modèles entraînés sur 2021–2024, testés sur 2025–avril 2026
        - Les performances peuvent se dégrader si le climat évolue significativement
          par rapport à la période d'entraînement (changement climatique)
        """)

    # ── Footer documentation ──────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:20px;color:#888;font-size:12px;">
        PollenGuard · Projet Data Science · Rennes 2026<br>
        Données : Open-Meteo API · Air Quality API · Réseau Sentinelles<br>
        Modèles entraînés sur Python 3.13 · scikit-learn · XGBoost
    </div>
    """, unsafe_allow_html=True)
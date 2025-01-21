import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des performances des étudiants",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Titre de l'application
st.title("📊 Prédiction des Performances des Étudiants")
st.markdown(
    """
    Cette application utilise un modèle de régression linéaire pour prédire les performances des étudiants en fonction de différents paramètres. 
    Veuillez remplir le formulaire ci-dessous pour tester le modèle.
    """
)

# Charger les données
data = pd.read_csv("Student_Performance (1).csv")

# Renommer les colonnes
data = data.rename(
    columns={
        "Hours Studied": "HS",
        "Previous Scores": "Scores",
        "Extracurricular Activities": "Activities",
        "Sleep Hours": "sommeil",
        "Sample Question Papers Practiced": "pratique",
        "Performance Index": "Performance",
    }
)

# Convertir les activités en valeurs numériques
data["Activities"] = data["Activities"].map({"Yes": 1, "No": 0})

# Séparer les données
X = data.drop("Performance", axis=1)
y = data["Performance"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Formulaire utilisateur
st.sidebar.header("Paramètres d'entrée")
hours_studied = st.sidebar.number_input("Heures d'étude", min_value=0.0, max_value=24.0, value=5.0)
previous_scores = st.sidebar.number_input("Scores précédents", min_value=0.0, max_value=100.0, value=70.0)
activities = st.sidebar.radio("Participe à des activités extrascolaires ?", ["Yes", "No"])
sleep_hours = st.sidebar.number_input("Heures de sommeil", min_value=0.0, max_value=24.0, value=8.0)
practice_papers = st.sidebar.number_input(
    "Nombre de questionnaires pratiqués", min_value=0, max_value=50, value=5
)

# Convertir les données saisies
activities_numeric = 1 if activities == "Yes" else 0
input_data = pd.DataFrame(
    {
        "HS": [hours_studied],
        "Scores": [previous_scores],
        "Activities": [activities_numeric],
        "sommeil": [sleep_hours],
        "pratique": [practice_papers],
    }
)

# Faire des prédictions
if st.sidebar.button("Prédire"):
    prediction = model.predict(input_data)
    st.success(f"✨ Indice de performance prédit : **{prediction[0]:.2f}**")
    
# Évaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Évaluation du modèle")
st.write(f"**Erreur quadratique moyenne (MSE):** {mse:.2f}")
st.write(f"**Coefficient de détermination (R²):** {r2:.2f}")

# Affichage des données
st.subheader("Aperçu des données")
st.dataframe(data)

# Personnalisation des couleurs
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stSidebar {
        background-color: #f4f4f4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

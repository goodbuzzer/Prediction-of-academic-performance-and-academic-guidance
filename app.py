import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Charger le dataset
df = pd.read_excel('dataset.xlsx',index_col=0)

# Calculer les indicateurs
num_rows = df.shape[0]
note_min = df["moyenne"].min()
note_max = df["moyenne"].max()
data_types = df.dtypes.value_counts()

# Configuration de la page
st.set_page_config(page_title="Tableau de Bord Dataset", layout="wide")

# Ajouter une image et la centrer
st.image('Logo AH.jpg',width=500)

# Titre de l'application
st.title('Projet B-Connect🔵')
st.title("Analyse prédictive des resultats scolaires et aide à l'orientation scolaire")

st.markdown("""
<style>
.indicator-container {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 10px;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
}
.indicator-title {
    font-size: 20px;
    color: #333;
}
.indicator-value {
    font-size: 32px;
    font-weight: bold;
    color: #007BFF;
}
.indicator-icon {
    font-size: 40px;
    color: #007BFF;
}
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.header('Quelques indicateurs')

# Utilisation de conteneurs pour styliser les indicateurs
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="indicator-container">
        <div class="indicator-icon">👨‍🎓</div>
        <div class="indicator-title">Effectif</div>
        <div class="indicator-value">{}</div>
    </div>
    """.format(num_rows), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="indicator-container">
        <div class="indicator-icon">🥇</div>
        <div class="indicator-title">Meilleure Moyenne</div>
        <div class="indicator-value">{}</div>
    </div>
    """.format(note_max), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="indicator-container">
        <div class="indicator-icon">⚠️</div>
        <div class="indicator-title">Pire Moyenne</div>
        <div class="indicator-value">{}</div>
    </div>
    """.format(note_min), unsafe_allow_html=True)


# Afficher un aperçu du dataset
st.header('Aperçu du Dataset')
st.dataframe(df.head())


# Sélectionner les features et la target
features = st.multiselect('Sélectionnez les features', df.columns.tolist())
target = st.selectbox('Sélectionnez la target', df.columns.tolist())

if features and target:
    X = df[features]
    y = df[target]

    # Fractionner les données
    st.header('Fractionner les données')
    test_size = st.slider('Taille du test set (%)', 10, 50, 20) / 100
    random_state = st.number_input('Random State', 0, 100, 32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Définir les modèles à évaluer
    # models = {
    #     'Régression Linéaire': LinearRegression(),
    #     'Random Forest': RandomForestRegressor()
    #     # Ajoutez d'autres modèles ici si nécessaire
    # }
    models = {
    'Régression Linéaire': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Support Vector Regression (SVR)': SVR(),
    'Decision Tree': DecisionTreeRegressor(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    # Ajoutez d'autres modèles ici si nécessaire
    }

    # Entraîner et évaluer les modèles
    best_model_name = None
    best_r2_score = -float('inf')

    for model_name, model in models.items():
        # st.header(f'Entraînement du modèle: {model_name}')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # st.write(f"R² Score pour {model_name}: {r2}")

        # Sélectionner le meilleur modèle basé sur le R² Score
        if r2 > best_r2_score:
            best_r2_score = r2
            best_model_name = model_name

    # st.header(f"Meilleur modèle sélectionné: {best_model_name}")

    # Prédire avec le meilleur modèle sélectionné
    st.header('Prédictions de vos performances 📘')

    # Laisser l'utilisateur entrer les valeurs pour la prédiction
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"Entrez la valeur pour {feature}", value=0.0)

    # Créer un DataFrame avec les valeurs d'entrée utilisateur
    user_df = pd.DataFrame([user_input])

    # Utiliser le meilleur modèle pour la prédiction
    if best_model_name:
        best_model = models[best_model_name]
        best_model.fit(X, y)  # Réentraîner sur toutes les données (optionnel, selon le cas d'utilisation)
        user_prediction = best_model.predict(user_df)

        # Afficher le résultat de la prédiction avec style
        st.subheader(f"Prédiction du pourcentage de la variable '{target}':")
        st.markdown(f"<p style='font-size:20px;font-weight:bold;color:#00b4d8;'>{user_prediction[0]}</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="indicator-container">
                <div class="indicator-icon">🚀</div>
                <div class="indicator-title">{}</div>
                <div class="indicator-value">{:.2f}{}</div>
            </div>
            """.format(target, user_prediction[0]*100,' %'), unsafe_allow_html=True)
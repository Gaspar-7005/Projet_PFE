import streamlit as st
import joblib
import pandas as pd 
import numpy as np
import plotly.express as px 

# chargement du modele
model = joblib.load("best_model.pkl")
#fontion de prediction 
def make_prediction(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability*100, 2)
    return prediction, probability


st.title("Application de prediction de Défault de paiement ")
st.write("cette application utilise un model de machine learning pour predire si un client sera en defaut de paiement ou pas")
st.sidebar.header("Information sur le client")

#Saisie des caracteristiques du client 
limit_bal = st.sidebar.number_input("Montant du credit", min_value=0, value=50000)
sex = st.sidebar.selectbox("sexe du client", ["Female","Male"])
education = st.sidebar.selectbox("Niveau d'education", ['Graduate school','University', 'High school', 'Others'])
marriage = st.sidebar.selectbox("satut matrimonial du client", ['Single','Married','Others'])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)


#Saisie des stauts de paiement 
payment_status_sep = st.sidebar.selectbox("Statut de paiement en septembre", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], index=2)
payment_status_aug = st.sidebar.selectbox("Statut de paiement en Aout", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], index=2)
payment_status_jul = st.sidebar.selectbox("Statut de paiement en Juillet", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], index=2)
payment_status_jun= st.sidebar.selectbox("Statut de paiement en Juin", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], index=2)
payment_status_may = st.sidebar.selectbox("Statut de paiement en Mai", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], index=2)
payment_status_apr = st.sidebar.selectbox("Statut de paiement en Avril", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], index=2)

# relevés de facturation 
bill_statement_sep = st.sidebar.number_input("Relevé de facturation en septembre", min_value=0, value=10000)
bill_statement_aug = st.sidebar.number_input("Relevé de facturation en Aout", min_value=0, value=10000)
bill_statement_jul = st.sidebar.number_input("Relevé de facturation en juillet", min_value=0, value=10000)
bill_statement_jun = st.sidebar.number_input("Relevé de facturation en juin", min_value=0, value=10000)
bill_statement_may = st.sidebar.number_input("Relevé de facturation en Mai", min_value=0, value=10000)
bill_statement_apr = st.sidebar.number_input("Relevé de facturation en avril", min_value=0, value=10000)

# saisie des paiement precedent
previous_payment_sep  = st.sidebar.number_input("paiement précédent en septembre", min_value=0, value=5000)
previous_payment_aug  = st.sidebar.number_input("paiement précédent en Aout", min_value=0, value=5000)
previous_payment_jul  = st.sidebar.number_input("paiement précédent en Juillet", min_value=0, value=5000)
previous_payment_jun  = st.sidebar.number_input("paiement précédent en Juin", min_value=0, value=5000)
previous_payment_may  = st.sidebar.number_input("paiement précédent en Mai", min_value=0, value=5000)
previous_payment_apr  = st.sidebar.number_input("paiement précédent en Avril", min_value=0, value=5000)

# Creer un dataframe avec les caracteriqtique renseignées
input_data = pd.DataFrame({
    'limit_bal': [limit_bal],
    'sex' : [sex],
    'education': [education],
    'marriage' : [marriage],
    'age': [age],
    'payment_status_sep': [payment_status_sep],
    'payment_status_aug': [payment_status_aug],
    'payment_status_jul': [payment_status_jul],
    'payment_status_jun': [payment_status_aug],
    'payment_status_may': [payment_status_aug],
    'payment_status_apr': [payment_status_jul],
    'bill_statement_sep': [bill_statement_sep],
    'bill_statement_aug': [bill_statement_aug],
    'bill_statement_jul': [bill_statement_jul],
    'bill_statement_jun': [bill_statement_jun],
    'bill_statement_may': [bill_statement_may],
    'bill_statement_apr': [bill_statement_apr],
    'previous_payment_sep':[previous_payment_sep],
    'previous_payment_aug':[previous_payment_aug],
    'previous_payment_jul':[previous_payment_jul],
    'previous_payment_jun':[previous_payment_jun],
    'previous_payment_may':[previous_payment_may],
    'previous_payment_apr':[previous_payment_apr]

})

# st.dataframe(input_data)

#Prediction
if st.sidebar.button("Prédire"):
    prediction, probability = make_prediction(input_data)
    st.subheader("probabilités : ")
    prob_df = pd.DataFrame({
        'Categories': ["No Default","Default"],
        'Probabilité': probability[0]
    })
    st.dataframe(prob_df)
    

    fig  = px.bar(prob_df, x= 'Categories', y='Probabilité', labels={'Probabilité': 'Probabilité (%)'})
    st.plotly_chart(fig)
    st.subheader(" Résultat de la Prédiction:")
    if prediction[0]== 1:
        st.error('Le client sera en default de paiement.')
    else:
        st.success('Le client ne sera pas en default de paiement.')



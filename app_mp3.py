import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from feature_extractor import extract_features_from_mp3

# --- Sidinställningar (måste vara första streamlit-kommandot) ---
st.set_page_config(
    page_title="Hit eller Flopp? 🎵",
    page_icon="🎶",
    layout="centered"
)

# --- Ladda Modeller och Data (en gång) ---
@st.cache_resource
def load_model_assets():
    try:
        model = tf.keras.models.load_model('spotify_hit_predictor.h5')
        scaler = joblib.load('scaler.pkl')
        features_list = joblib.load('features.pkl')
        return model, scaler, features_list
    except FileNotFoundError:
        st.error("🚨 Modellfilerna hittades inte! Se till att `train_model.py` har körts.")
        return None, None, None

model, scaler, features_list = load_model_assets()

# --- NYTT: Cache-funktion för ljudanalys ---
# Denna "decorator" ser till att om samma fil-innehåll laddas upp igen,
# återanvänds resultatet istället för att analysera om filen.
@st.cache_data
def get_cached_features(file_content):
    # Skapa en temporär fil i minnet för librosa att läsa
    temp_path = "temp_audio_for_cache.mp3"
    with open(temp_path, "wb") as f:
        f.write(file_content)
    
    features = extract_features_from_mp3(temp_path)
    os.remove(temp_path)
    return features

# --- NYTT: Funktioner för visualisering ---
def plot_radar_chart(song_features, feature_names):
    # Hårdkodade genomsnittsvärden för en typisk hit/flopp (baserat på analys av datasetet)
    avg_hit_features = {'Danceability': 0.68, 'Energy': 0.75, 'Loudness': -5.5, 'Speechiness': 0.08, 'Acousticness': 0.15, 'Instrumentalness': 0.01, 'Liveness': 0.18, 'Valence': 0.55, 'Tempo': 122}
    avg_flop_features = {'Danceability': 0.55, 'Energy': 0.60, 'Loudness': -8.5, 'Speechiness': 0.10, 'Acousticness': 0.35, 'Instrumentalness': 0.05, 'Liveness': 0.20, 'Valence': 0.45, 'Tempo': 118}
    
    # Skala om loudness och tempo för att passa på samma 0-1 skala som de flesta andra
    for features in [song_features, avg_hit_features, avg_flop_features]:
        features['Loudness'] = (features['Loudness'] + 60) / 60
        features['Tempo'] = (features['Tempo'] - 40) / 180

    fig = go.Figure()

    # Lägg till data för "Din Låt"
    fig.add_trace(go.Scatterpolar(
        r=[song_features[f] for f in feature_names],
        theta=feature_names, fill='toself', name='Din Låt', line_color='gold'
    ))
    # Lägg till data för "Typisk Hit"
    fig.add_trace(go.Scatterpolar(
        r=[avg_hit_features[f] for f in feature_names],
        theta=feature_names, fill='toself', name='Typisk Hit', line_color='lightgreen', opacity=0.6
    ))
    # Lägg till data för "Typisk Flopp"
    fig.add_trace(go.Scatterpolar(
        r=[avg_flop_features[f] for f in feature_names],
        theta=feature_names, fill='toself', name='Typisk Flopp', line_color='salmon', opacity=0.6
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, title="Jämförelse av Ljudegenskaper"
    )
    return fig

def plot_feature_importance(model, feature_names):
    first_layer_weights = model.layers[0].get_weights()[0]
    importance = np.mean(np.abs(first_layer_weights), axis=1)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)
    fig = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h', marker_color='#1DB954'))
    fig.update_layout(title="Vilka egenskaper tycker modellen är viktigast?")
    return fig


# --- HUVUDAPPLIKATION ---
if model: # Kör bara appen om modellen laddades korrekt
    st.title("🎵 Hit eller Flopp? Predictor")
    st.markdown("Ladda upp en **MP3-fil** och låt ett neuralt nätverk avgöra om din låt har potential att bli en hit!")
    
    uploaded_file = st.file_uploader("Dra och släpp din låt här", type=["mp3"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        
        with st.spinner('🎧 Analyserar ljudvågorna... (Första gången kan ta en stund)'):
            # NYTT: Använder cache-funktionen
            extracted_features = get_cached_features(uploaded_file.getvalue())

        if extracted_features:
            st.success("✅ Analys klar!")
            
            input_df = pd.DataFrame([extracted_features])[features_list]
            input_scaled = scaler.transform(input_df)
            prediction_proba = model.predict(input_scaled)[0][0]
            
            st.markdown("---")
            st.header("✨ Resultat ✨")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Mätare för sannolikhet
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction_proba * 100,
                    title={'text': "Sannolikhet för Hit (%)", 'font': {'size': 20}},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#1DB954"},
                           'steps': [{'range': [0, 50], 'color': '#F08080'}, {'range': [50, 100], 'color': '#90EE90'}]}
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.markdown("### Bedömning")
                if prediction_proba > 0.5:
                    st.success(f"**HIT!**\n\nSannolikhet: **{prediction_proba:.1%}**")
                    st.balloons()
                else:
                    st.error(f"**FLOPP!**\n\nSannolikhet: **{prediction_proba:.1%}**")
                    st.snow()
            
            st.markdown("---")
            st.header("💡 Djupare Insikter")
            
            # NYTT: Flikar för att organisera diagrammen
            tab1, tab2 = st.tabs(["Jämför din låt (Radar)", "Vad tycker modellen är viktigt?"])

            with tab1:
                st.plotly_chart(plot_radar_chart(extracted_features.copy(), features_list), use_container_width=True)
                with st.expander("Vad betyder detta diagram?"):
                    st.markdown("""
                    Detta **radar-diagram** visar din låts ljudprofil (`Din Låt`) i jämförelse med en genomsnittlig `Typisk Hit` och `Typisk Flopp`.
                    - Ju mer din låts form liknar en **Hit**, desto större är chansen för en positiv förutsägelse.
                    - Egenskaper som sticker ut kan ge en ledtråd till varför modellen ger ett visst resultat.
                    *(Notera: Vissa värden som Loudness och Tempo har skalats om för att passa i diagrammet).*
                    """)
            with tab2:
                st.plotly_chart(plot_feature_importance(model, features_list), use_container_width=True)
                with st.expander("Vad betyder detta diagram?"):
                    st.markdown("""
                    Detta diagram visar vilka ljudegenskaper som modellens första lager anser vara viktigast **generellt sett**.
                    - En egenskap med en längre stapel har större påverkan på modellens beslut.
                    - Detta visar inte hur viktig en egenskap var för just *din* låt, utan hur modellen har "lärt sig" att väga olika faktorer.
                    """)

        else:
            st.error("❌ Kunde inte analysera ljudfilen. Försök med en annan fil.")
    else:
        st.info("Väntar på att en MP3-fil ska laddas upp...")
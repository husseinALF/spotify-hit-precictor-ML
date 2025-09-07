import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Säkerställ att filen finns
if not os.path.exists('Spotify_Youtube.csv'):
    print("Fel: 'Spotify_Youtube.csv' hittades inte.")
    print("Ladda ner datasetet från https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube/data och placera det i samma mapp.")
else:
    # 1. Ladda och Rensa Data
    print("Läser in datasetet...")
    df = pd.read_csv('Spotify_Youtube.csv')

    # Välj relevanta audiofunktioner och målvariabel
    features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
        'Instrumentalness', 'Liveness', 'Valence', 'Tempo'
    ]
    target = 'Stream'
    df_clean = df[features + [target]].dropna()

    # 2. Förbered Data
    print("Förbereder data...")
    # Skapa en binär målvariabel: 1 för "Hit" (topp 20% av streams), 0 för "Flop"
    hit_threshold = df_clean[target].quantile(0.80)
    df_clean['Hit'] = (df_clean[target] > hit_threshold).astype(int)

    X = df_clean[features]
    y = df_clean['Hit']

    # Dela upp data i tränings- och testset, stratifiera för att behålla klassbalansen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisera funktioner (anpassa på träningsdata, transformera båda)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Hantera Obalanserade Klasser
    # Beräkna klassvikter för att ge mer vikt åt den underrepresenterade "Hit"-klassen
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Använder klassvikter för att hantera obalans: {class_weights_dict}")

    # 4. Bygg Modellen (med skydd mot överanpassning)
    print("Bygger TensorFlow-modellen...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),  # Dropout-lager för att förhindra överanpassning
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),  # Ett till Dropout-lager
        tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid för binär klassificering (sannolikhet)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # 5. Träna Modellen
    print("Tränar modellen...")
    # EarlyStopping avbryter träningen när valideringsförlusten slutar förbättras
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=1
    )

    # 6. Utvärdera Modellen
    print("Utvärderar modellens prestanda...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"\nResultat på testdata - Förlust: {loss:.4f}, Träffsäkerhet: {accuracy:.4f}")

    # 6b. Avancerad Utvärdering
print("\n--- Avancerad Utvärdering ---")

# Gör förutsägelser på testdatan
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype("int32")

# Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Flop (0)', 'Hit (1)']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Flop', 'Hit'], yticklabels=['Flop', 'Hit'])
plt.xlabel('Förutsagd klass')
plt.ylabel('Verklig klass')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') # Sparar bilden
plt.show()

# AUC-ROC Kurva
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nAUC (Area Under Curve): {roc_auc:.4f}")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png') # Sparar bilden
plt.show()

# 7. Spara Modell och Skalare
print("Sparar modell, skalare och funktionslista...")
model.save('spotify_hit_predictor.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')

print("\nKlart! Modellfilerna har skapats.")

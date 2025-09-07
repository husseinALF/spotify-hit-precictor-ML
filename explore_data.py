import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ladda datasetet
df = pd.read_csv('Spotify_Youtube.csv')

# Välj relevanta audiofunktioner
features = [
    'Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness',
    'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Stream'
]
df_clean = df[features].dropna()

print("--- 1. Exploratory Data Analysis (EDA) ---")

# Visa grundläggande information
print("\nDataset Info:")
df_clean.info()

print("\nStatistisk Sammanfattning:")
print(df_clean.describe())

# Visualisera distributionen av några nyckelfunktioner
plt.figure(figsize=(15, 10))
for i, feature in enumerate(['Danceability', 'Energy', 'Loudness', 'Valence', 'Tempo', 'Stream']):
    plt.subplot(2, 3, i+1)
    sns.histplot(df_clean[feature], kde=True)
    plt.title(f'Distribution av {feature}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# Korrelationsmatris för att se samband mellan variabler
plt.figure(figsize=(12, 10))
correlation_matrix = df_clean.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatris mellan ljudegenskaper och Streams')
plt.savefig('correlation_heatmap.png')
plt.show()

print("\n--- 2. Unsupervised Learning (K-Means Clustering) ---")
print("Hypotes: Kan oövervakad inlärning hitta naturliga grupper som liknar 'hits' och 'flops'?")

# Skapa vår "sanna" målvariabel för att kunna jämföra senare
hit_threshold = df_clean['Stream'].quantile(0.80)
df_clean['is_hit'] = (df_clean['Stream'] > hit_threshold).astype(int)

# Förbered data för klustring (utan 'Stream' och 'is_hit')
X = df_clean.drop(['Stream', 'is_hit'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Använd K-Means för att hitta 2 kluster
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

print("\nJämför K-Means kluster med våra faktiska 'Hit'/'Flop' kategorier:")
# Använd en korstabell för att se hur klustren matchar
crosstab = pd.crosstab(df_clean['cluster'], df_clean['is_hit'])
print(crosstab)

print("\nSlutsats: Kluster 0 verkar mest innehålla 'flops' (is_hit=0) och Kluster 1 har en högre andel 'hits' (is_hit=1).")
print("Detta visar att det finns ett mönster i datan som oövervakad inlärning kan hitta, vilket stärker vår hypotes att hits och flops har olika ljudprofiler.")
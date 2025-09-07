import librosa
import numpy as np
from sklearn.preprocessing import minmax_scale

def extract_features_from_mp3(audio_path):
    """
    Analyserar en MP3-fil och extraherar/approximerar de ljudegenskaper
    som vår modell förväntar sig.
    
    Args:
        audio_path (str): Sökvägen till MP3-filen.
        
    Returns:
        dict: En dictionary med de 9 extraherade egenskaperna.
    """
    try:
        # Ladda ljudfilen. 'sr=None' behåller originalets samplingsfrekvens.
        y, sr = librosa.load(audio_path, sr=None)

        # 1. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # 2. Energy / Loudness
        # Vi använder RMS (Root-Mean-Square) energi som en proxy.
        rms = librosa.feature.rms(y=y)
        energy = np.mean(rms)
        # Loudness i dB är logaritmisk. Detta är en approximation.
        loudness = librosa.amplitude_to_db(rms, ref=np.max).mean()

        # 3. Danceability
        # Mycket svår att approximera. En enkel proxy är en kombination av tempo
        # och beat-regelbundenhet. Vi skalar det till 0-1.
        beat_frames = librosa.beat.beat_track(y=y, sr=sr)[1]
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_consistency = 1 - np.std(np.diff(beat_times))
        # Normalisera tempot (t.ex. 120 bpm är ett vanligt danstempo)
        tempo_factor = np.exp(-0.5 * ((tempo - 120) / 20) ** 2)
        danceability = minmax_scale([beat_consistency * tempo_factor], feature_range=(0, 1))[0]

        # 4. Valence (musikalisk positivitet)
        # Enormt komplex. Proxy: ett "ljust" spektralt centrum och snabbare tempo
        # associeras ofta med gladare musik.
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        valence = minmax_scale([spectral_centroid * tempo_factor], feature_range=(0, 1))[0]

        # 5. Acousticness
        # Mycket svår. Proxy: Hög zero-crossing rate kan indikera brist på tonalitet (brus).
        # Låg harmonisk-till-perkussiv-ratio kan indikera mer akustiska element.
        h, p = librosa.effects.hpss(y)
        acousticness = np.mean(h) / (np.mean(p) + 1e-6) # Undvik division med noll
        acousticness = minmax_scale([acousticness], feature_range=(0, 1))[0]

        # 6. Instrumentalness
        # Nästan omöjlig utan en dedikerad modell för sångseparation.
        # Vi kan använda en proxy baserad på spektral "flathet".
        # Musik med sång har ofta en mindre "platt" spektralprofil.
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        instrumentalness = minmax_scale([spectral_flatness], feature_range=(0, 1))[0]

        # 7. Speechiness
        # Ofta associerat med specifika frekvensband och rytm. En enkel proxy:
        # Leta efter låg energi och hög spektral bandbredd.
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        speechiness = (energy < 0.1) and (spectral_bandwidth > 2000)
        speechiness = minmax_scale([speechiness * spectral_bandwidth], feature_range=(0, 1))[0]

        # 8. Liveness
        # Omöjlig att detektera på ett tillförlitligt sätt. Vi sätter ett neutralt standardvärde
        # baserat på medelvärdet från det ursprungliga datasetet.
        liveness = 0.18 # Neutralt värde

        # Samla allt i en dictionary
        features = {
            'Danceability': float(danceability),
            'Energy': float(minmax_scale([energy])[0]), # Skala energin till 0-1
            'Loudness': float(loudness),
            'Speechiness': float(speechiness),
            'Acousticness': float(acousticness),
            'Instrumentalness': float(instrumentalness),
            'Liveness': float(liveness),
            'Valence': float(valence),
            'Tempo': float(tempo)
        }
        
        return features

    except Exception as e:
        print(f"Ett fel inträffade vid analys av ljudfilen: {e}")
        return None
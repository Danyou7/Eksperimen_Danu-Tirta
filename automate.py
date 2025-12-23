import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing(input_file, output_file):
    print(f"Memuat data dari {input_file}...")
    df = pd.read_csv(input_file)

    num_features = [
        'Temperature', 'Humidity', 'Wind Speed', 
        'Precipitation (%)', 'Atmospheric Pressure', 
        'UV Index', 'Visibility (km)'
    ]
    cat_features = ['Cloud Cover', 'Season', 'Location']
    target = 'Weather Type'

    print("Menangani outliers dengan Winsorization...")
    for col in num_features:
        df[col] = winsorize(df[col], limits=[0.05, 0.05])

    print("Melakukan encoding pada fitur kategorikal dan target...")
    le = LabelEncoder()
    
    df[target] = le.fit_transform(df[target])
    
    for col in cat_features:
        df[col] = le.fit_transform(df[col])

    print("Melakukan normalisasi data numerik dengan StandardScaler...")
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    df.to_csv(output_file, index=False)
    print(f"Proses selesai! Data tersimpan di: {output_file}")

if __name__ == "__main__":
    input_csv = "weather_classification_data.csv"
    output_csv = "preprocessed.csv"
    
    try:
        run_preprocessing(input_csv, output_csv)
    except FileNotFoundError:
        print(f"Error: File '{input_csv}' tidak ditemukan. Pastikan file dataset berada di folder yang sama.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
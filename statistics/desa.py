import pandas as pd

# Muat dataset yang sudah bersih
try:
    df = pd.read_csv('../data/stunting_bersih.csv')
except FileNotFoundError:
    print("File 'stunting_bersih.csv' tidak ditemukan. Jalankan script cleansing terlebih dahulu.")
    exit()

# --- Metode 1: Menggunakan value_counts() (Paling Sederhana) ---
print("--- Hasil menggunakan .value_counts() ---")
jumlah_per_desa_vc = df['desa'].value_counts()
print(jumlah_per_desa_vc)


# --- Metode 2: Menggunakan groupby().size() (Lebih Fleksibel) ---
print("\n--- Hasil menggunakan .groupby().size() ---")
jumlah_per_desa_gb = df.groupby('desa').size().sort_values(ascending=False)
print(jumlah_per_desa_gb)
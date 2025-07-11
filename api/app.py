import joblib
import pandas as pd

# 1. Memuat semua model yang dibutuhkan
rf_model = joblib.load('../random_forest_model.pkl')
kmeans_model = joblib.load('../kmeans_model.pkl')
scaler = joblib.load('../scaler.pkl')

# 2. Contoh ada data anak baru
data_anak_baru = {
    'usia': 12,
    'berat': 7.5,
    'tinggi': 72.1,
    'lila': 13.1,
    'rt': 3,
    'rw': 5,
    'zs_bb_u': -2.5,
    'zs_tb_u': -3.1,
    'zs_bb_tb': -1.9,
    'jk_P': 1 # Misal jenis kelamin perempuan
}
df_baru = pd.DataFrame([data_anak_baru])

# ==========================================================
# PENGGUNAAN MODEL RANDOM FOREST UNTUK PREDIKSI STUNTING
# ==========================================================
# Pastikan urutan kolom sama seperti saat training
kolom_rf = ['usia', 'rt', 'rw', 'berat', 'tinggi', 'lila', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb', 'jk_P']
prediksi_stunting = rf_model.predict(df_baru[kolom_rf])

# Mengubah hasil prediksi (misal 0) kembali ke label (misal 'Stunting')
# Anda perlu menyimpan LabelEncoder (le) atau mappingnya jika ingin hasil berupa teks
label_mapping = {0: 'Stunting', 1: 'Tidak Stunting'}
print(f"Hasil Prediksi Stunting: {label_mapping[prediksi_stunting[0]]}")


# ==========================================================
# PENGGUNAAN MODEL K-MEANS UNTUK MENENTUKAN CLUSTER
# ==========================================================
# Pilih fitur yang relevan untuk clustering
kolom_cluster = ['usia', 'berat', 'tinggi', 'lila', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb']
data_untuk_cluster = df_baru[kolom_cluster]

# Skalakan data baru menggunakan scaler yang sudah disimpan!
data_scaled = scaler.transform(data_untuk_cluster)

# Prediksi cluster
prediksi_cluster = kmeans_model.predict(data_scaled)
print(f"Anak ini masuk ke dalam Cluster: {prediksi_cluster[0]}")
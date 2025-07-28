import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. PERSIAPAN DATA
print(">>> 1. Memuat dan Mempersiapkan Data...")
df = pd.read_csv('./data/final_stunting_recalculated.csv')
print("Data awal berhasil dimuat.")

df_processed = pd.get_dummies(df, columns=['jk'], drop_first=True) # Hanya encode jk

le = LabelEncoder()
df_processed['status_stunting'] = le.fit_transform(df['status_stunting']) # Gunakan df asli untuk target
print("\nMapping Label Status Stunting:", {label: index for index, label in enumerate(le.classes_)})

# --- PERUBAHAN UTAMA DI SINI ---
y = df_processed['status_stunting']

# Definisikan fitur dasar yang ingin kita gunakan (tanpa Z-score)
fitur_dasar = ['usia', 'berat', 'tinggi']
# Ambil kolom hasil one-hot encoding dari 'jk'
fitur_kategorikal = [col for col in df_processed.columns if col.startswith('jk_')] # Hanya ambil fitur jk

# Gabungkan semua fitur final yang akan digunakan untuk melatih model
fitur_final = fitur_dasar + fitur_kategorikal
X = df_processed[fitur_final]
print(f"\nModel akan dilatih dengan {len(X.columns)} fitur.")
# --- PERUBAHAN SELESAI ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. KLASIFIKASI DENGAN RANDOM FOREST
print("\n>>> 2. Melatih Model Klasifikasi Random Forest...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%") # Akurasi sekarang akan lebih realistis
print("\nLaporan Klasifikasi (Multikelas):")
print(classification_report(y_test, y_pred, target_names=le.classes_))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (Multikelas)')
plt.ylabel('Kelas Aktual')
plt.xlabel('Kelas Prediksi')
plt.show()

print("\n>>> Menganalisis Faktor Paling Berpengaruh...")
importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values(by='Pentingnya', ascending=False)
print("Peringkat 3 Fitur Teratas:")
print(feature_importance_df.head(3))
plt.figure(figsize=(10, 8))
sns.barplot(x='Pentingnya', y='Fitur', data=feature_importance_df.head(3))
plt.title('Tingkat Pengaruh 3 Fitur Teratas')
plt.xlabel('Tingkat Pengaruh (Importance)')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()

# 3. CLUSTERING
print("\n>>> 3. Melakukan Analisis Clustering dengan K-Means...")
cluster_features = ['usia', 'berat', 'tinggi', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb']
X_cluster = df[cluster_features] # Clustering tetap pakai z-score untuk analisis profil
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)


wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_test = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10).fit(X_cluster_scaled)
    wcss.append(kmeans_test.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid(True)
plt.show()




optimal_k = 4 # Optimal k berdasarkan Elbow Method
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
print(f"\nData berhasil dikelompokkan ke dalam {optimal_k} cluster.")
print("\nKarakteristik Rata-rata per Cluster:")
print(df.groupby('cluster')[cluster_features].mean())
print("\nDistribusi Status Stunting Rinci per Cluster:")
print(df.groupby('cluster')['status_stunting'].value_counts(normalize=True).unstack().fillna(0))


# ==============================================================================
# 2.1 VISUALISASI SEBARAN CLUSTER (SCATTER PLOT)
# ==============================================================================
print("\n>>> 2.1 Membuat Visualisasi Sebaran Cluster...")

plt.figure(figsize=(12, 8))

# Membuat scatter plot menggunakan seaborn
# hue='cluster' akan memberi warna titik berdasarkan nomor clusternya
# palette='viridis' adalah pilihan skema warna, bisa diganti
sns.scatterplot(
    data=df,
    x='zs_tb_u',
    y='zs_bb_tb',
    hue='cluster',
    palette='viridis',
    s=50,  # Ukuran titik
    alpha=0.7 # Transparansi titik
)

# Menambahkan garis referensi standar WHO
plt.axvline(x=-2, color='red', linestyle='--', label='Batas Stunting (zs_tb_u = -2)')
plt.axhline(y=-2, color='orange', linestyle='--', label='Batas Gizi Kurang (zs_bb_tb = -2)')
plt.axhline(y=2, color='purple', linestyle='--', label='Batas Gizi Lebih (zs_bb_tb = +2)')

plt.title('Visualisasi Sebaran Cluster Anak Berdasarkan Status Gizi', fontsize=16)
plt.xlabel('Z-Score Tinggi Badan / Umur (Indikator Stunting)')
plt.ylabel('Z-Score Berat Badan / Tinggi Badan (Indikator Kurus/Gempal)')
plt.grid(True)
plt.legend()
plt.show()


# ==============================================================================
# 3.5 MEMBUAT PROFIL DOMINAN PER DESA UNTUK PEMETAAN
# ==============================================================================
print("\n>>> 3.5 Menghitung Profil Dominan per Desa untuk Pemetaan...")

# Menghitung jumlah anak per cluster di setiap desa menggunakan crosstab
profil_desa = pd.crosstab(df['desa_kel'], df['cluster'])

# Mencari cluster dengan jumlah anak terbanyak (dominan) di setiap desa
profil_desa['profil_dominan'] = profil_desa.idxmax(axis=1)

print("\n--- Tabel Profil Dominan per Desa (Data untuk Peta) ---")
print(profil_desa)

# Menyimpan hasil profil desa ke file CSV
output_dir_peta = 'data'
os.makedirs(output_dir_peta, exist_ok=True)
output_file_peta = os.path.join(output_dir_peta, 'profil_dominan_desa.csv')
profil_desa.to_csv(output_file_peta)

print(f"\nData profil dominan per desa berhasil disimpan di: '{output_file_peta}'")


# --- KODE TAMBAHAN SETELAH MEMBUAT PROFIL DESA ---

# Menghitung proporsi/persentase setiap cluster di setiap desa
# normalize='index' akan menghitung persentase per baris (per desa)
profil_persentase = pd.crosstab(df['desa_kel'], df['cluster'], normalize='index')

# Mengubahnya menjadi format persen yang mudah dibaca
profil_persentase = (profil_persentase * 100).round(2)

# Mengganti nama kolom agar lebih jelas
profil_persentase = profil_persentase.add_prefix('persen_cluster_')

print("\n--- Tabel Persentase Komposisi Cluster per Desa ---")
print(profil_persentase)

# Menyimpan hasil persentase ini ke file CSV terpisah
profil_persentase.to_csv('data/profil_persentase_desa.csv')
print("\nData persentase komposisi per desa berhasil disimpan di: 'data/profil_persentase_desa.csv'")

# ==============================================================================
# 2.3 MENYIMPAN RINGKASAN CLUSTER
# ==============================================================================


# Ambil data karakteristik rata-rata
cluster_summary_means = df.groupby('cluster')[cluster_features].mean()
# Ambil data distribusi status stunting
cluster_summary_dist = df.groupby('cluster')['status_stunting'].value_counts(normalize=True).unstack().fillna(0)

# Simpan ke file CSV di folder 'data'
output_dir_summary = 'data'
os.makedirs(output_dir_summary, exist_ok=True)
cluster_summary_means.to_csv(os.path.join(output_dir_summary, 'cluster_summary_means.csv'))
cluster_summary_dist.to_csv(os.path.join(output_dir_summary, 'cluster_summary_distribution.csv'))

print("Data ringkasan cluster berhasil disimpan.")

# 4. MENYIMPAN MODEL
print("\n>>> 4. Menyimpan Model Final...")
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)
joblib.dump(rf_classifier, os.path.join(output_dir, 'rf_realistis.pkl'))
joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_final.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler_final.pkl'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder_multiclass.pkl'))
model_columns = list(X.columns)
joblib.dump(model_columns, os.path.join(output_dir, 'model_columns_realistis.pkl'))
print(f"Semua model (realistis) berhasil disimpan di folder '{output_dir}/'")


df.to_csv('data/data_lengkap_teranalisis.csv', index=False)
print("DataFrame lengkap dengan cluster berhasil disimpan.")
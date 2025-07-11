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

# ==============================================================================
# 1. PERSIAPAN DATA
# ==============================================================================
print(">>> 1. Memuat dan Mempersiapkan Data...")

# Memuat data hasil data engineering
file_path = './data/final_stunting_recalculated.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
    exit()

print("Data awal berhasil dimuat.")

# --- Pra-pemrosesan untuk Machine Learning ---

# a. Mengubah semua kolom kategorikal yang relevan menjadi angka
# One-Hot Encoding untuk fitur yang tidak memiliki urutan
df_processed = pd.get_dummies(df, columns=['jk', 'desa_kel', 'status_gizi'], drop_first=True)

# b. Mengubah kolom target 'status_stunting' menjadi angka
le = LabelEncoder()
df_processed['status_stunting'] = le.fit_transform(df_processed['status_stunting'])
print("\nMapping Label Stunting:", {label: index for index, label in enumerate(le.classes_)})

# c. Memisahkan Fitur (X) dan Target (y)
# y adalah target prediksi kita
y = df_processed['status_stunting']
# X adalah semua kolom lain kecuali kolom target dan kolom yang tidak relevan (seperti tanggal)
X = df_processed.drop(columns=['status_stunting', 'tgl_lahir', 'tanggal_pengecekan'])

# d. Membagi data menjadi data latih (training) dan data uji (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nUkuran data latih: {X_train.shape}")
print(f"Ukuran data uji: {X_test.shape}")

# ==============================================================================
# 2. KLASIFIKASI DENGAN RANDOM FOREST
# ==============================================================================
print("\n>>> 2. Melatih Model Klasifikasi Random Forest...")

# Inisialisasi dan melatih model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)

# Membuat prediksi dan evaluasi
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Kelas Aktual')
plt.xlabel('Kelas Prediksi')
plt.show()

# Visualisasi Faktor Paling Berpengaruh
print("\n>>> Menganalisis Faktor Paling Berpengaruh...")
importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values(by='Pentingnya', ascending=False)
print("Peringkat 10 Fitur Teratas:")
print(feature_importance_df.head(10))
plt.figure(figsize=(10, 8))
sns.barplot(x='Pentingnya', y='Fitur', data=feature_importance_df.head(15))
plt.title('Tingkat Pengaruh 15 Fitur Teratas')
plt.xlabel('Tingkat Pengaruh (Importance)')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. CLUSTERING DENGAN K-MEANS
# ==============================================================================
print("\n>>> 3. Melakukan Analisis Clustering dengan K-Means...")

# Memilih fitur numerik inti untuk clustering
cluster_features = ['usia_terkoreksi', 'berat', 'tinggi', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb']
# Mengganti 'usia' menjadi 'usia_terkoreksi' jika nama kolomnya itu
if 'usia' in df.columns:
    cluster_features[0] = 'usia'

X_cluster = df[cluster_features]
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Elbow Method untuk mencari K optimal
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

# Menerapkan K-Means dengan K optimal (misal: 3)
optimal_k = 3
print(f"\nBerdasarkan Elbow Method, kita pilih k = {optimal_k} untuk clustering.")
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
print(f"Data berhasil dikelompokkan ke dalam {optimal_k} cluster.")

# Analisis hasil cluster
print("\nKarakteristik Rata-rata per Cluster:")
print(df.groupby('cluster')[cluster_features].mean())
print("\nDistribusi Status Stunting per Cluster:")
print(df.groupby('cluster')['status_stunting'].value_counts(normalize=True).unstack().fillna(0))

# ==============================================================================
# 4. MENYIMPAN MODEL FINAL
# ==============================================================================
print("\n>>> 4. Menyimpan Model Final...")
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

joblib.dump(rf_classifier, os.path.join(output_dir, 'random_forest_final.pkl'))
joblib.dump(kmeans, os.path.join(output_dir, 'kmeans_final.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler_final.pkl'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder_stunting.pkl'))

print(f"Semua model berhasil disimpan di folder '{output_dir}/'")
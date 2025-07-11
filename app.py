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

file_path = 'data/final_stunting.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
    exit()

# Menampilkan beberapa baris pertama untuk memastikan data termuat
print("Data awal yang dimuat:")
print(df.head())

# --- Pra-pemrosesan untuk Machine Learning ---

# a. Mengubah semua kolom kategorikal menjadi angka (One-Hot Encoding)
# Kolom 'jk', 'desa', dan 'status_gizi' akan diubah menjadi kolom numerik
df_processed = pd.get_dummies(df, columns=['jk', 'desa', 'status_gizi'], drop_first=True)
print("\nData setelah One-Hot Encoding (beberapa kolom):")
print(df_processed.head())

# b. Mengubah kolom target 'status_stunting' menjadi angka (Label Encoding)
le = LabelEncoder()
# Pastikan kolom target ada sebelum encoding
if 'status_stunting' in df_processed.columns:
    df_processed['status_stunting'] = le.fit_transform(df_processed['status_stunting'])
    print("\nMapping Label 'status_stunting':", {label: index for index, label in enumerate(le.classes_)})
else:
    print("Error: Kolom target 'status_stunting' tidak ditemukan.")
    exit()

# c. Memisahkan Fitur (X) dan Target (y)
# y adalah apa yang ingin kita prediksi ('status_stunting')
y = df_processed['status_stunting']
# X adalah semua kolom lain yang akan digunakan sebagai prediktor
X = df_processed.drop(columns=['status_stunting'])

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

# Membuat prediksi pada data uji dan mengevaluasi performa
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Menampilkan Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Kelas Aktual')
plt.xlabel('Kelas Prediksi')
plt.show()

# Menganalisis Faktor Paling Berpengaruh (Feature Importance)
print("\n>>> Menganalisis Faktor Paling Berpengaruh...")
importances = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances})
feature_importance_df = feature_importance_df.sort_values(by='Pentingnya', ascending=False)

print("Peringkat Fitur Berdasarkan Tingkat Pengaruh:")
print(feature_importance_df.head(10)) # Tampilkan 10 fitur teratas

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

# Memilih fitur numerik yang relevan untuk clustering (tanpa 'lila')
cluster_features = ['usia', 'berat', 'tinggi', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb']
X_cluster = df[cluster_features]

# Menormalisasi data
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Menentukan jumlah cluster (k) yang optimal dengan Metode Siku (Elbow Method)
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Menerapkan K-Means dengan k=3 (atau sesuaikan dengan hasil Elbow Method Anda)
optimal_k = 3
print(f"\nBerdasarkan Elbow Method, kita pilih k = {optimal_k} untuk clustering.")
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)
df['cluster'] = cluster_labels
print(f"Data berhasil dikelompokkan ke dalam {optimal_k} cluster.")

# Menganalisis karakteristik setiap cluster
print("\nKarakteristik Rata-rata per Cluster:")
print(df.groupby('cluster')[cluster_features].mean())

# Menganalisis distribusi status stunting di setiap cluster
stunting_distribution = df.groupby('cluster')['status_stunting'].value_counts(normalize=True).unstack().fillna(0)
print("\nDistribusi Status Stunting per Cluster:")
print(stunting_distribution)


# ==============================================================================
# 4. MENYIMPAN MODEL FINAL
# ==============================================================================
print("\n>>> 4. Menyimpan Model Final...")

# Membuat folder 'models' jika belum ada
os.makedirs('models', exist_ok=True)

joblib.dump(rf_classifier, 'models/random_forest_final.pkl')
joblib.dump(kmeans, 'models/kmeans_final.pkl')
joblib.dump(scaler, 'models/scaler_final.pkl')
joblib.dump(le, 'models/label_encoder_stunting.pkl')

print("Semua model berhasil disimpan di folder 'models/'.")
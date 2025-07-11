import pandas as pd
import numpy as np

# Ganti './data/stunting.csv' dengan nama file Anda yang sebenarnya
file_path = './data/stunting.csv' 

# ==============================================================================
# 1. MEMUAT DATA
# ==============================================================================
try:
    df = pd.read_csv(file_path)
    print(">>> File CSV berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
    exit()

# ==============================================================================
# 2. INSPEKSI AWAL
# ==============================================================================
print("\n>>> 2. Inspeksi Awal Data")
print("Informasi DataFrame:")
df.info()
print("\nJumlah nilai kosong per kolom (sebelum cleansing):")
print(df.isnull().sum())

# ==============================================================================
# 3. MENANGANI NILAI KOSONG (MISSING VALUES)
# ==============================================================================
print("\n>>> 3. Menangani Nilai Kosong...")

# Daftar kolom numerik dan kategorikal
numeric_cols = ['rt', 'rw', 'berat', 'tinggi', 'lila', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb']
categorical_cols = ['jk', 'desa_kel', 'status_gizi', 'stunting', 'nama_ortu']

# Isi nilai kosong pada kolom numerik dengan median
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"- Nilai kosong di '{col}' diisi dengan median ({median_value}).")

# Isi nilai kosong pada kolom kategorikal dengan modus (nilai paling sering muncul)
for col in categorical_cols:
    if col in df.columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
        print(f"- Nilai kosong di '{col}' diisi dengan modus ('{mode_value}').")

# ==============================================================================
# 4. MEMPERBAIKI TIPE DATA
# ==============================================================================
print("\n>>> 4. Memperbaiki Tipe Data...")

# Konversi tanggal
if 'tgl_lahir' in df.columns:
    df['tgl_lahir'] = pd.to_datetime(df['tgl_lahir'], errors='coerce')
    print("- Kolom 'tgl_lahir' dikonversi ke tipe datetime.")

# Pastikan tipe data numerik sudah benar
for col in ['rt', 'rw']:
    if col in df.columns:
        df[col] = df[col].astype(int)
print("- Kolom 'rt' dan 'rw' dikonversi ke tipe integer.")

# ==============================================================================
# 5. MENANGANI DUPLIKAT
# ==============================================================================
print("\n>>> 5. Menangani Duplikat...")
jumlah_duplikat = df.duplicated().sum()
if jumlah_duplikat > 0:
    df.drop_duplicates(inplace=True)
    print(f"- {jumlah_duplikat} baris duplikat telah dihapus.")
else:
    print("- Tidak ada data duplikat.")

# ==============================================================================
# 6. STANDARISASI DATA KATEGORIKAL
# ==============================================================================
print("\n>>> 6. Standarisasi Data Kategorikal...")

# Standarisasi kolom 'jk'
if 'jk' in df.columns:
    df['jk'] = df['jk'].astype(str).str.strip().str.upper().str[0]
    df['jk'] = df['jk'].replace({'W': 'P'})
    df = df[df['jk'].isin(['L', 'P'])] # Hanya pertahankan L atau P
    print(f"- Kolom 'jk' distandarisasi. Nilai unik: {df['jk'].unique()}")

# Standarisasi kolom teks lainnya (menjadi huruf kapital)
for col in ['desa_kel', 'status_gizi', 'stunting']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip()
        print(f"- Kolom '{col}' distandarisasi (uppercase).")

# ==============================================================================
# 7. FINALISASI & MENYIMPAN DATA BERSIH
# ==============================================================================
print("\n>>> 7. Finalisasi dan Menyimpan Data...")

# Hapus kolom identifier yang tidak diperlukan untuk pemodelan
kolom_tidak_perlu = ['nama', 'nama_ortu']
df.drop(columns=kolom_tidak_perlu, inplace=True)
print(f"- Kolom {kolom_tidak_perlu} dihapus.")

# Simpan ke file CSV baru
output_file = './data/stunting_bersih_final.csv'
df.to_csv(output_file, index=False)
print(f"\nProses cleansing selesai! Data bersih disimpan di: '{output_file}'")

# Tampilkan info dan 5 baris pertama data bersih
print("\nInformasi data setelah cleansing:")
df.info()
print("\nContoh data bersih:")
print(df.head())
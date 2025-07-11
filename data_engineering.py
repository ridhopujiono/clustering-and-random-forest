import pandas as pd
import numpy as np
from pygrowup_erknet import Calculator
from pandas.tseries.offsets import DateOffset

# ==============================================================================
# 1. SETUP DAN FUNGSI
# ==============================================================================
print(">>> 1. Setup dan persiapan fungsi...")

# Inisialisasi Kalkulator
calc = Calculator(adjust_height_data=False, include_cdc=False)

def tebak_usia_bulan(jk, tinggi_cm, berat_kg, target_zs_tb_u):
    """
    Menebak usia anak (dalam bulan) dengan mencari Z-score TB/U
    yang paling mendekati nilai target.
    
    Catatan: Parameter berat_kg saat ini tidak digunakan dalam fungsi ini.
    """
    sex = 'M' if jk.upper() == 'L' else 'F'
    usia_bulan_tebakan = None
    selisih_terkecil = float('inf')
    
    # Iterasi dari usia 0 hingga 60 bulan
    for usia_bulan in range(0, 61):
        try:
            # Hitung Z-score TB/U untuk usia yang sedang diuji
            zscore = calc.lhfa(measurement=tinggi_cm, age_in_months=usia_bulan, sex=sex)
            selisih = abs(zscore - target_zs_tb_u)
            
            # Jika selisihnya lebih kecil dari yang pernah ditemukan, simpan usia ini
            if selisih < selisih_terkecil:
                selisih_terkecil = selisih
                usia_bulan_tebakan = usia_bulan
        except:
            # Lanjutkan jika ada error (misal, tinggi di luar jangkauan untuk usia tsb)
            continue
            
    return usia_bulan_tebakan # Kita hanya butuh usianya

# ==============================================================================
# 2. MEMUAT DAN MEMPROSES DATA
# ==============================================================================
print(">>> 2. Memuat data bersih...")
try:
    # Ganti dengan nama file hasil cleansing Anda jika berbeda
    df = pd.read_csv('stunting_bersih.csv') 
    # Pastikan kolom tanggal lahir adalah tipe datetime
    df['tgl lahir'] = pd.to_datetime(df['tgl lahir'], errors='coerce')
    df.dropna(subset=['tgl lahir'], inplace=True) # Hapus baris jika tgl lahir tidak valid
except FileNotFoundError:
    print("Error: File 'stunting_bersih.csv' tidak ditemukan.")
    exit()
except KeyError:
    print("Error: Pastikan file CSV memiliki kolom 'jk', 'tinggi', 'berat', 'tgl lahir', dan 'zs_tb_u'.")
    exit()

# ==============================================================================
# 3. MEMBUAT KOLOM BARU
# ==============================================================================
print(">>> 3. Membuat kolom 'usia_terkoreksi' dan 'tanggal_pengecekan'...")

# --- Membuat kolom usia baru menggunakan fungsi tebak_usia_bulan ---
# Menerapkan fungsi ke setiap baris data
# Note: Ini mungkin butuh waktu beberapa saat jika datanya besar
df['usia_terkoreksi'] = df.apply(
    lambda row: tebak_usia_bulan(
        jk=row['jk'], 
        tinggi_cm=row['tinggi'], 
        berat_kg=row['berat'], 
        target_zs_tb_u=row['zs_tb_u']
    ),
    axis=1
)
print("- Kolom 'usia_terkoreksi' berhasil dibuat.")

# Hapus baris di mana usia tidak bisa ditebak
df.dropna(subset=['usia_terkoreksi'], inplace=True)
df['usia_terkoreksi'] = df['usia_terkoreksi'].astype(int)

# --- Membuat kolom tanggal pengecekan ---
# Menambahkan usia (dalam bulan) ke tanggal lahir
df['tanggal_pengecekan_dt'] = df.apply(
    lambda row: row['tgl lahir'] + DateOffset(months=row['usia_terkoreksi']),
    axis=1
)

# Mapping nama bulan dari Inggris ke Indonesia
bulan_map = {
    'January': 'Januari', 'February': 'Februari', 'March': 'Maret', 'April': 'April',
    'May': 'Mei', 'June': 'Juni', 'July': 'Juli', 'August': 'Agustus',
    'September': 'September', 'October': 'Oktober', 'November': 'November', 'December': 'Desember'
}

# Format tanggal menjadi "Bulan Tahun" dan terjemahkan
df['bulan_pengecekan_en'] = df['tanggal_pengecekan_dt'].dt.strftime('%B')
df['tahun_pengecekan'] = df['tanggal_pengecekan_dt'].dt.strftime('%Y')
df['bulan_pengecekan_id'] = df['bulan_pengecekan_en'].map(bulan_map)
df['tanggal_pengecekan'] = df['bulan_pengecekan_id'] + ' ' + df['tahun_pengecekan']
print("- Kolom 'tanggal_pengecekan' berhasil dibuat.")

# ==============================================================================
# 4. FINALISASI DAN MENYIMPAN DATA
# ==============================================================================
print(">>> 4. Finalisasi dan menyimpan data...")

# Memilih dan mengatur urutan kolom untuk file final
kolom_final = [
    'jk', 'tgl lahir', 'berat', 'tinggi', 
    'usia_terkoreksi', 'tanggal_pengecekan',
    'zs_bb_u', 'zs_tb_u', 'zs_bb_tb', 
    'status_gizi', 'status_stunting', 'desa'
]
# Ambil hanya kolom yang ada di dataframe awal untuk menghindari error
kolom_tersedia = [kol for kol in kolom_final if kol in df.columns]
df_final = df[kolom_tersedia]


# Membuat folder 'data' jika belum ada
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'final_stunting_recalculated.csv')

# Menyimpan DataFrame final ke CSV
df_final.to_csv(output_file, index=False)
print(f"\nProses selesai! Data final berhasil disimpan di: '{output_file}'")
print("\nLima baris pertama data final:")
print(df_final.head())
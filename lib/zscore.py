# lib/zscore.py

from pygrowup_erknet import Calculator

# Inisialisasi Kalkulator
calc = Calculator(adjust_height_data=False, include_cdc=False)

def hitung_zscore_balita(jk, usia_bulan, berat_kg, tinggi_cm):
    """
    Menghitung Z-score untuk BALITA (0-60 bulan) menggunakan pygrowup-erknet.
    """
    if not (0 <= usia_bulan <= 60):
        return None
    try:
        sex = 'M' if jk.upper() == 'L' else 'F'
        
        zs_bb_u = calc.wfa(measurement=berat_kg, age_in_months=usia_bulan, sex=sex)
        zs_tb_u = calc.lhfa(measurement=tinggi_cm, age_in_months=usia_bulan, sex=sex)
        
        if usia_bulan < 24:
            zs_bb_tb = calc.wfl(measurement=berat_kg, sex=sex, length=tinggi_cm)
        else:
            zs_bb_tb = calc.wfh(measurement=berat_kg, sex=sex, height=tinggi_cm)
            
        hasil = {
            'zs_bb_u': round(zs_bb_u, 2),
            'zs_tb_u': round(zs_tb_u, 2),
            'zs_bb_tb': round(zs_bb_tb, 2)
        }
        return hasil
    except Exception:
        return None
    

def tebak_usia_bulan(jk, tinggi_cm, berat_kg, target_zs_tb_u, tolerance=0.2):
    sex = 'M' if jk.upper() == 'L' else 'F'
    usia_bulan_tebakan = None
    selisih_terkecil = float('inf')
    
    for usia_bulan in range(0, 61):
        try:
            zscore = calc.lhfa(measurement=tinggi_cm, age_in_months=usia_bulan, sex=sex)
            selisih = abs(zscore - target_zs_tb_u)
            if selisih < selisih_terkecil:
                selisih_terkecil = selisih
                usia_bulan_tebakan = usia_bulan
        except:
            continue
        
    return usia_bulan_tebakan, selisih_terkecil



# jk = input("Jenis Kelamin (L/P): ")
# tinggi_cm = float(input("Tinggi (cm): "))
# berat_kg = float(input("Berat (kg): "))
# target_zs_tb_u = float(input("Z-score TB/U Target: "))

# usia_bulan_prediksi, selisih = tebak_usia_bulan(jk, tinggi_cm, berat_kg, target_zs_tb_u)

# print(f"Perkiraan usia balita: {usia_bulan_prediksi} bulan (selisih Z-score: {round(selisih, 2)})")


# if usia_bulan_prediksi is not None:
#     print(f"Perkiraan usia balita: {usia_bulan_prediksi} bulan")
#     zscore = hitung_zscore_balita(jk, usia_bulan_prediksi, berat_kg, tinggi_cm)
#     print(f"Hasil Z-score pada usia tersebut: {zscore}")
# else:
#     print("Gagal memprediksi usia.")


# # input from terminal
# # jk = input("Jenis Kelamin (L/P): ")
# # usia_bulan = int(input("Usia (bulan): "))
# # berat_kg = float(input("Berat (kg): "))
# # tinggi_cm = float(input("Tinggi (cm): "))

# # jk = input("Jenis Kelamin (L/P): ")
# # usia_bulan = float(input("Usia (bulan): "))
# # berat_kg = float(input("Berat (kg): "))
# # tinggi_cm = float(input("Tinggi (cm): "))

# # # hitung zscore
# # zscore = hitung_zscore_balita(jk, usia_bulan, berat_kg, tinggi_cm)
# # print(zscore)

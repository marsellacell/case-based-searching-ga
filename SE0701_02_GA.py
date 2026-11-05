# =====================================================================
# Case Based Searching - Kecerdasan Buatan
# Algoritma Genetika (GA) untuk mencari nilai MINIMUM fungsi:
#
#   f(x1, x2) = -( sin(x1)*cos(x2)*tan(x1+x2) + 0.5*exp(1 - sqrt(x2^2)) )
#
#   dengan domain: -10 <= x1 <= 10  dan  -10 <= x2 <= 10
#
# Semua proses GA (inisialisasi, decode, fitness, seleksi orangtua,
# crossover, mutasi, pergantian generasi) dibangun SENDIRI tanpa library GA.
# Library yang dipakai hanya 'random' (bilangan acak) dan 'math' (sin, cos, dll).
# =====================================================================

import random
import math

# ---------------------------------------------------------------------
# PARAMETER GA
# ---------------------------------------------------------------------
UKURAN_POPULASI = 50     # jumlah kromosom dalam satu populasi
BIT_PER_VARIABEL = 10    # jumlah gen (bit) untuk mengkodekan satu variabel
JUMLAH_VARIABEL = 2      # variabel yang dicari: x1 dan x2
PANJANG_KROMOSOM = BIT_PER_VARIABEL * JUMLAH_VARIABEL  # total gen = 20
BATAS_BAWAH = -10        # rb : batas bawah domain x1 dan x2
BATAS_ATAS = 10          # ra : batas atas domain x1 dan x2
PC = 0.8                 # probabilitas crossover (pindah silang)
PM = 0.05                # probabilitas mutasi per gen (~1/panjang kromosom)
UKURAN_TURNAMEN = 3      # jumlah kandidat yang diadu pada tournament selection
JUMLAH_ELIT = 2          # jumlah kromosom terbaik yang langsung lolos ke generasi berikutnya
MAKS_GENERASI = 100      # kriteria berhenti: jumlah generasi maksimum
SEED = None              # isi angka (misal 42) agar hasil bisa diulang sama persis


# ---------------------------------------------------------------------
# FUNGSI OBJEKTIF
# ---------------------------------------------------------------------
def fungsi_objektif(x1, x2):
    """Menghitung nilai f(x1, x2) sesuai soal (nilai ini yang ingin diminimalkan)."""
    bagian_1 = math.sin(x1) * math.cos(x2) * math.tan(x1 + x2)  # sin(x1)cos(x2)tan(x1+x2)
    bagian_2 = 0.5 * math.exp(1 - math.sqrt(x2 ** 2))           # 1/2 * exp(1 - sqrt(x2^2))
    return -(bagian_1 + bagian_2)                               # tanda minus di depan sesuai soal


# ---------------------------------------------------------------------
# 1. INISIALISASI POPULASI
# ---------------------------------------------------------------------
def buat_kromosom():
    """Membuat satu kromosom biner acak sepanjang PANJANG_KROMOSOM, contoh: [1,0,1,1,...]."""
    return [random.randint(0, 1) for _ in range(PANJANG_KROMOSOM)]


def inisialisasi_populasi():
    """Membuat populasi awal berisi UKURAN_POPULASI kromosom acak."""
    return [buat_kromosom() for _ in range(UKURAN_POPULASI)]


# ---------------------------------------------------------------------
# 2. DEKODE KROMOSOM
# ---------------------------------------------------------------------
def dekode_gen(gen):
    """
    Mengubah potongan gen biner menjadi bilangan real dalam domain [rb, ra]
    menggunakan rumus representasi biner:

        x = rb + (ra - rb) / (sum 2^-i) * (sum g_i * 2^-i),   i = 1..N
    """
    pembilang = 0.0   # sum g_i * 2^-i
    penyebut = 0.0    # sum 2^-i
    for i, g in enumerate(gen, start=1):   # i dimulai dari 1 sesuai rumus
        pembilang += g * (2 ** -i)
        penyebut += 2 ** -i
    return BATAS_BAWAH + (BATAS_ATAS - BATAS_BAWAH) * pembilang / penyebut


def dekode_kromosom(kromosom):
    """Memecah kromosom: 10 gen pertama -> x1, 10 gen berikutnya -> x2, lalu didekode."""
    gen_x1 = kromosom[:BIT_PER_VARIABEL]
    gen_x2 = kromosom[BIT_PER_VARIABEL:]
    return dekode_gen(gen_x1), dekode_gen(gen_x2)


# ---------------------------------------------------------------------
# 3. PERHITUNGAN FITNESS
# ---------------------------------------------------------------------
def hitung_fitness(kromosom):
    """
    Karena masalahnya MINIMASI, fitness = -f(x1, x2).
    Semakin kecil f, semakin besar fitness, sehingga kromosom tersebut semakin baik.
    (Rumus 1/(f+a) tidak dipakai karena f bisa bernilai negatif sangat besar akibat tan.)
    """
    x1, x2 = dekode_kromosom(kromosom)
    return -fungsi_objektif(x1, x2)


# ---------------------------------------------------------------------
# 4. PEMILIHAN ORANGTUA (Tournament Selection)
# ---------------------------------------------------------------------
def pilih_orangtua(populasi, daftar_fitness):
    """
    Ambil UKURAN_TURNAMEN kromosom secara acak, lalu pilih yang fitness-nya
    paling tinggi sebagai orangtua. Metode ini tetap bekerja walaupun fitness negatif.
    """
    kandidat = random.sample(range(len(populasi)), UKURAN_TURNAMEN)  # indeks kandidat acak
    pemenang = max(kandidat, key=lambda idx: daftar_fitness[idx])    # fitness tertinggi menang
    return populasi[pemenang]


# ---------------------------------------------------------------------
# 5. CROSSOVER (Single-Point Crossover)
# ---------------------------------------------------------------------
def crossover(ortu1, ortu2):
    """
    Dengan peluang PC, kedua orangtua ditukar bagiannya setelah satu titik potong acak.
    Contoh titik potong = 3:
        ortu1 = 101|0110  ->  anak1 = 101|1001
        ortu2 = 010|1001  ->  anak2 = 010|0110
    Jika tidak terjadi crossover, anak = salinan orangtua.
    """
    if random.random() < PC:
        titik = random.randint(1, PANJANG_KROMOSOM - 1)  # titik potong, bukan di ujung
        anak1 = ortu1[:titik] + ortu2[titik:]
        anak2 = ortu2[:titik] + ortu1[titik:]
        return anak1, anak2
    return ortu1[:], ortu2[:]   # [:] untuk menyalin list agar orangtua tidak ikut berubah


# ---------------------------------------------------------------------
# 6. MUTASI (Bit-Flip Mutation)
# ---------------------------------------------------------------------
def mutasi(kromosom):
    """Setiap gen punya peluang PM untuk dibalik nilainya (0 -> 1 atau 1 -> 0)."""
    hasil = kromosom[:]
    for i in range(PANJANG_KROMOSOM):
        if random.random() < PM:
            hasil[i] = 1 - hasil[i]   # membalik bit
    return hasil


# ---------------------------------------------------------------------
# 7. PERGANTIAN GENERASI (Generational Replacement + Elitisme)
# ---------------------------------------------------------------------
def ganti_generasi(populasi, daftar_fitness):
    """
    Membentuk populasi baru:
    - JUMLAH_ELIT kromosom terbaik disalin langsung (elitisme) agar solusi terbaik tidak hilang.
    - Sisanya diisi anak hasil seleksi orangtua, crossover, dan mutasi.
    """
    # urutkan indeks dari fitness tertinggi ke terendah, lalu ambil elit
    urutan = sorted(range(len(populasi)), key=lambda idx: daftar_fitness[idx], reverse=True)
    populasi_baru = [populasi[idx][:] for idx in urutan[:JUMLAH_ELIT]]

    # isi sisa populasi dengan anak-anak baru
    while len(populasi_baru) < UKURAN_POPULASI:
        ortu1 = pilih_orangtua(populasi, daftar_fitness)
        ortu2 = pilih_orangtua(populasi, daftar_fitness)
        anak1, anak2 = crossover(ortu1, ortu2)
        populasi_baru.append(mutasi(anak1))
        if len(populasi_baru) < UKURAN_POPULASI:   # jaga agar ukuran populasi tetap
            populasi_baru.append(mutasi(anak2))

    return populasi_baru


# ---------------------------------------------------------------------
# PROGRAM UTAMA (LOOP EVOLUSI)
# ---------------------------------------------------------------------
def jalankan_ga():
    if SEED is not None:
        random.seed(SEED)

    # tampilkan parameter yang dipakai (untuk screenshot laporan)
    print("=" * 65)
    print("  ALGORITMA GENETIKA - MINIMASI f(x1, x2)")
    print("=" * 65)
    print(f"  Ukuran populasi    : {UKURAN_POPULASI}")
    print(f"  Panjang kromosom   : {PANJANG_KROMOSOM} bit ({BIT_PER_VARIABEL} bit per variabel)")
    print(f"  Seleksi orangtua   : Tournament (k = {UKURAN_TURNAMEN})")
    print(f"  Crossover          : Single-point, Pc = {PC}")
    print(f"  Mutasi             : Bit-flip, Pm = {PM}")
    print(f"  Pergantian generasi: Generational + elitisme ({JUMLAH_ELIT} elit)")
    print(f"  Kriteria berhenti  : {MAKS_GENERASI} generasi")
    print("=" * 65)

    populasi = inisialisasi_populasi()          # 1. inisialisasi
    kromosom_terbaik = None
    fitness_terbaik = float("-inf")

    for generasi in range(1, MAKS_GENERASI + 1):
        # 2 & 3. dekode dan hitung fitness seluruh kromosom
        daftar_fitness = [hitung_fitness(k) for k in populasi]

        # simpan kromosom terbaik sepanjang evolusi
        idx_terbaik = max(range(UKURAN_POPULASI), key=lambda idx: daftar_fitness[idx])
        if daftar_fitness[idx_terbaik] > fitness_terbaik:
            fitness_terbaik = daftar_fitness[idx_terbaik]
            kromosom_terbaik = populasi[idx_terbaik][:]

        # tampilkan progres setiap 10 generasi
        if generasi == 1 or generasi % 10 == 0:
            x1, x2 = dekode_kromosom(kromosom_terbaik)
            print(f"  Generasi {generasi:3d} | f terbaik = {fungsi_objektif(x1, x2):14.6f} "
                  f"| x1 = {x1:8.4f} | x2 = {x2:8.4f}")

        # 4-7. seleksi, crossover, mutasi, pergantian generasi
        if generasi < MAKS_GENERASI:
            populasi = ganti_generasi(populasi, daftar_fitness)

    # OUTPUT AKHIR sesuai permintaan soal
    x1, x2 = dekode_kromosom(kromosom_terbaik)
    print("=" * 65)
    print("  HASIL AKHIR")
    print("=" * 65)
    print(f"  Kromosom terbaik : {''.join(map(str, kromosom_terbaik))}")
    print(f"     gen x1        : {''.join(map(str, kromosom_terbaik[:BIT_PER_VARIABEL]))}")
    print(f"     gen x2        : {''.join(map(str, kromosom_terbaik[BIT_PER_VARIABEL:]))}")
    print(f"  Nilai x1         : {x1:.6f}")
    print(f"  Nilai x2         : {x2:.6f}")
    print(f"  Nilai f(x1, x2)  : {fungsi_objektif(x1, x2):.6f}")
    print("=" * 65)


if __name__ == "__main__":
    jalankan_ga()
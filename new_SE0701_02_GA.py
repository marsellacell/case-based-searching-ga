# Case Based Searching - Kecerdasan Buatan
# Kelompok 02 - SE0701
# Marsella Dwi Julianti (2311104004)
# Muhammad Rifqi Al Baqi Ananta (2311104005)
#
# Mencari nilai minimum fungsi f(x1, x2) dari soal, dengan x1 dan x2 dari -10 sampai 10,
# menggunakan Algoritma Genetika.
# Proses GA dibuat sendiri, library yang dipakai hanya random dan math.

import random
import math

# parameter GA
UKURAN_POPULASI = 50  # jumlah kromosom dalam satu populasi
BIT_PER_VARIABEL = 10  # jumlah bit untuk satu variabel
PANJANG_KROMOSOM = BIT_PER_VARIABEL * 2  # x1 dan x2, total 20 gen
BATAS_BAWAH = -10  # rb, batas bawah domain
BATAS_ATAS = 10  # ra, batas atas domain
PC = 0.8  # probabilitas crossover
PM = 0.05  # probabilitas mutasi per gen
UKURAN_TURNAMEN = 3  # jumlah kandidat di tournament selection
JUMLAH_ELIT = 2  # kromosom terbaik yang langsung masuk generasi berikutnya
MAKS_GENERASI = 100  # kriteria berhenti 1: generasi maksimum
BATAS_STAGNASI = 50  # kriteria berhenti 2: berhenti kalau tidak membaik selama 50 generasi
SEED = 42  # supaya hasil bisa diulang, isi None kalau mau acak


def fungsi_objektif(x1, x2):
    # rumus f(x1, x2) sesuai soal
    bagian_1 = math.sin(x1) * math.cos(x2) * math.tan(x1 + x2)
    bagian_2 = 0.5 * math.exp(1 - math.sqrt(x2 ** 2))
    return -(bagian_1 + bagian_2)


# 1. inisialisasi populasi
def buat_kromosom():
    # kromosom berisi 20 gen acak bernilai 0 atau 1
    return [random.randint(0, 1) for _ in range(PANJANG_KROMOSOM)]


def inisialisasi_populasi():
    return [buat_kromosom() for _ in range(UKURAN_POPULASI)]


# 2. dekode kromosom
def dekode_gen(gen):
    # rumus decode: rb + (ra - rb) / sum(2^-i) * sum(g_i * 2^-i)
    pembilang = 0.0  # sum(g_i * 2^-i)
    penyebut = 0.0  # sum(2^-i)
    for i, g in enumerate(gen, 1):  # i mulai dari 1 sesuai rumus
        pembilang += g * (2 ** -i)
        penyebut += 2 ** -i
    return BATAS_BAWAH + (BATAS_ATAS - BATAS_BAWAH) * pembilang / penyebut


def dekode_kromosom(kromosom):
    # 10 gen pertama untuk x1, 10 gen berikutnya untuk x2
    gen_x1 = kromosom[:BIT_PER_VARIABEL]
    gen_x2 = kromosom[BIT_PER_VARIABEL:]
    return dekode_gen(gen_x1), dekode_gen(gen_x2)


# 3. perhitungan fitness
def hitung_fitness(kromosom):
    # karena minimasi, fitness diambil dari negatif f, jadi makin kecil f makin besar fitness
    # 1/(f+a) tidak dipakai karena f bisa negatif sangat besar akibat tan
    x1, x2 = dekode_kromosom(kromosom)
    return -fungsi_objektif(x1, x2)


# 4. pemilihan orangtua (tournament selection)
def pilih_orangtua(populasi, daftar_fitness):
    # ambil 3 kromosom acak, yang fitness-nya paling tinggi jadi orangtua
    kandidat = random.sample(range(len(populasi)), UKURAN_TURNAMEN)
    pemenang = max(kandidat, key=lambda idx: daftar_fitness[idx])
    return populasi[pemenang]


# 5. crossover (single-point)
def crossover(ortu1, ortu2):
    # contoh titik potong 3: 101|0110 + 010|1001 -> 101|1001 dan 010|0110
    if random.random() < PC:
        titik = random.randint(1, PANJANG_KROMOSOM - 1)  # titik potong tidak di ujung
        anak1 = ortu1[:titik] + ortu2[titik:]
        anak2 = ortu2[:titik] + ortu1[titik:]
        return anak1, anak2
    return ortu1[:], ortu2[:]  # tidak crossover, anak hanya salinan orangtua


# 6. mutasi (bit-flip)
def mutasi(kromosom):
    hasil = kromosom[:]
    for i in range(PANJANG_KROMOSOM):
        if random.random() < PM:
            hasil[i] = 1 - hasil[i]  # balik bit 0 jadi 1, 1 jadi 0
    return hasil


# 7. pergantian generasi (generational + elitisme)
def ganti_generasi(populasi, daftar_fitness):
    # 2 kromosom terbaik langsung disalin supaya solusi terbaik tidak hilang
    urutan = sorted(range(len(populasi)), key=lambda idx: daftar_fitness[idx], reverse=True)
    populasi_baru = [populasi[idx][:] for idx in urutan[:JUMLAH_ELIT]]

    # sisa populasi diisi anak hasil seleksi, crossover, dan mutasi
    while len(populasi_baru) < UKURAN_POPULASI:
        ortu1 = pilih_orangtua(populasi, daftar_fitness)
        ortu2 = pilih_orangtua(populasi, daftar_fitness)
        anak1, anak2 = crossover(ortu1, ortu2)
        populasi_baru.append(mutasi(anak1))
        if len(populasi_baru) < UKURAN_POPULASI:   # jaga ukuran populasi tetap
            populasi_baru.append(mutasi(anak2))

    return populasi_baru


# input parameter dari user, tekan Enter untuk pakai default
def baca_input(label, default, tipe):
    teks = input(f"{label} [default {default}]: ").strip()
    if not teks:  # Enter saja, pakai default
        return default
    if teks.lower() == "none":  # untuk seed, ketik none supaya acak
        return None
    try:
        return tipe(teks)
    except ValueError:  # input salah, pakai default
        print(f"  input tidak valid, pakai default {default}")
        return default


def atur_parameter():
    global UKURAN_POPULASI, PC, PM, MAKS_GENERASI, SEED
    print("Input parameter GA (tekan Enter untuk pakai default)")
    # populasi minimal 3 supaya tournament bisa jalan
    UKURAN_POPULASI = max(UKURAN_TURNAMEN, baca_input("Ukuran populasi", UKURAN_POPULASI, int))
    # Pc dan Pm dibatasi 0 sampai 1
    PC = min(1.0, max(0.0, baca_input("Probabilitas crossover Pc (0-1)", PC, float)))
    PM = min(1.0, max(0.0, baca_input("Probabilitas mutasi Pm (0-1)", PM, float)))
    MAKS_GENERASI = max(1, baca_input("Jumlah generasi maksimum", MAKS_GENERASI, int))
    SEED = baca_input("Seed (ketik none supaya acak)", SEED, int)
    print()


def jalankan_ga():
    if SEED is not None:
        random.seed(SEED)

    # tampilkan parameter yang dipakai
    print("Algoritma Genetika - minimasi f(x1, x2)")
    print(f"Seed                : {SEED}")
    print(f"Ukuran populasi     : {UKURAN_POPULASI}")
    print(f"Panjang kromosom    : {PANJANG_KROMOSOM} bit ({BIT_PER_VARIABEL} bit per variabel)")
    print(f"Seleksi orangtua    : tournament, k {UKURAN_TURNAMEN}")
    print(f"Crossover           : single-point, Pc {PC}")
    print(f"Mutasi              : bit-flip, Pm {PM}")
    print(f"Pergantian generasi : generational + elitisme ({JUMLAH_ELIT} elit)")
    print(f"Kriteria berhenti   : maks {MAKS_GENERASI} generasi / stagnan {BATAS_STAGNASI} generasi")
    print()

    populasi = inisialisasi_populasi()
    kromosom_terbaik = None
    fitness_terbaik = float("-inf")
    tanpa_perbaikan = 0  # hitung berapa generasi berturut-turut tidak ada perbaikan

    for generasi in range(1, MAKS_GENERASI + 1):
        # dekode dan hitung fitness semua kromosom
        daftar_fitness = [hitung_fitness(k) for k in populasi]

        # simpan kromosom terbaik selama evolusi
        idx_terbaik = max(range(UKURAN_POPULASI), key=lambda idx: daftar_fitness[idx])
        if daftar_fitness[idx_terbaik] > fitness_terbaik:
            fitness_terbaik = daftar_fitness[idx_terbaik]
            kromosom_terbaik = populasi[idx_terbaik][:]
            tanpa_perbaikan = 0  # ada perbaikan, reset
        else:
            tanpa_perbaikan += 1

        # tampilkan progres tiap 10 generasi
        if generasi == 1 or generasi % 10 == 0:
            x1, x2 = dekode_kromosom(kromosom_terbaik)
            print(f"Generasi {generasi:3d} | f terbaik: {fungsi_objektif(x1, x2):12.6f} "
                  f"| x1: {x1:8.4f} | x2: {x2:8.4f}")

        # kriteria berhenti 2: stagnan
        if tanpa_perbaikan >= BATAS_STAGNASI:
            print(f"Berhenti di generasi {generasi}, tidak ada perbaikan selama {BATAS_STAGNASI} generasi")
            break

        # bentuk generasi baru (seleksi, crossover, mutasi, elitisme)
        if generasi < MAKS_GENERASI:
            populasi = ganti_generasi(populasi, daftar_fitness)

    # output akhir
    x1, x2 = dekode_kromosom(kromosom_terbaik)
    kode = "".join(map(str, kromosom_terbaik))
    print()
    print("Hasil akhir")
    print(f"Kromosom terbaik : {kode}")
    print(f"  gen x1         : {kode[:BIT_PER_VARIABEL]}")
    print(f"  gen x2         : {kode[BIT_PER_VARIABEL:]}")
    print(f"Nilai x1         : {x1:.6f}")
    print(f"Nilai x2         : {x2:.6f}")
    print(f"Nilai f(x1, x2)  : {fungsi_objektif(x1, x2):.6f}")


if __name__ == "__main__":
    atur_parameter()
    jalankan_ga()
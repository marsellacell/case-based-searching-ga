"""Case Based Searching - GA untuk meminimalkan
f(x1, x2) = -(sin(x1) cos(x2) tan(x1 + x2) + 0.5 exp(1 - sqrt(x2^2))),  -10 <= x1, x2 <= 10
Semua proses GA dibuat sendiri; hanya memakai random dan math."""

import math
import random

# ===== Parameter GA =====
UKURAN_POPULASI = 100
BIT_PER_VARIABEL = 16                        # resolusi ~ 20/65535 = 0.0003
PANJANG_KROMOSOM = 2 * BIT_PER_VARIABEL      # x1 dan x2
BATAS_BAWAH, BATAS_ATAS = -10, 10
PC = 0.8                                     # probabilitas crossover
PM = 1 / PANJANG_KROMOSOM                    # probabilitas mutasi per gen (~1 bit/kromosom)
UKURAN_TURNAMEN = 3
JUMLAH_ELIT = 2
MAKS_GENERASI = 200
BATAS_STAGNASI = 50                          # berhenti jika tidak membaik selama N generasi
EPS_TAN = 1e-3                               # |cos(x1+x2)| di bawah ini = singularitas tan
HUKUMAN = 1e9                                # nilai f untuk solusi di dekat singularitas
SEED = None                                  # isi angka agar hasil bisa diulang


def fungsi_objektif(x1, x2):
    """f(x1,x2). Dekat singularitas tan (cos ~ 0) nilainya meledak -> diberi hukuman."""
    if abs(math.cos(x1 + x2)) < EPS_TAN:
        return HUKUMAN
    return -(math.sin(x1) * math.cos(x2) * math.tan(x1 + x2)
             + 0.5 * math.exp(1 - math.sqrt(x2 ** 2)))


def inisialisasi_populasi():
    return [[random.randint(0, 1) for _ in range(PANJANG_KROMOSOM)]
            for _ in range(UKURAN_POPULASI)]


def dekode_gen(gen):
    """x = rb + (ra - rb) / sum(2^-i) * sum(g_i * 2^-i), i = 1..N"""
    pembilang = sum(g * 2 ** -i for i, g in enumerate(gen, start=1))
    penyebut = sum(2 ** -i for i in range(1, len(gen) + 1))
    return BATAS_BAWAH + (BATAS_ATAS - BATAS_BAWAH) * pembilang / penyebut


def dekode_kromosom(kromosom):
    return (dekode_gen(kromosom[:BIT_PER_VARIABEL]),
            dekode_gen(kromosom[BIT_PER_VARIABEL:]))


def hitung_fitness(kromosom):
    """Minimasi f, jadi fitness = -f (makin kecil f, makin besar fitness)."""
    return -fungsi_objektif(*dekode_kromosom(kromosom))


def pilih_orangtua(populasi, fitness):
    """Tournament selection: kandidat acak, fitness tertinggi menang."""
    kandidat = random.sample(range(len(populasi)), UKURAN_TURNAMEN)
    return populasi[max(kandidat, key=lambda i: fitness[i])]


def crossover(ortu1, ortu2):
    """Single-point crossover dengan peluang PC."""
    if random.random() < PC:
        titik = random.randint(1, PANJANG_KROMOSOM - 1)
        return ortu1[:titik] + ortu2[titik:], ortu2[:titik] + ortu1[titik:]
    return ortu1[:], ortu2[:]


def mutasi(kromosom):
    """Bit-flip: tiap gen dibalik dengan peluang PM."""
    return [1 - g if random.random() < PM else g for g in kromosom]


def ganti_generasi(populasi, fitness):
    """Generational replacement dengan elitisme."""
    urutan = sorted(range(len(populasi)), key=lambda i: fitness[i], reverse=True)
    baru = [populasi[i][:] for i in urutan[:JUMLAH_ELIT]]
    while len(baru) < UKURAN_POPULASI:
        anak1, anak2 = crossover(pilih_orangtua(populasi, fitness),
                                 pilih_orangtua(populasi, fitness))
        baru.append(mutasi(anak1))
        if len(baru) < UKURAN_POPULASI:
            baru.append(mutasi(anak2))
    return baru


def jalankan_ga(cetak=True):
    if SEED is not None:
        random.seed(SEED)

    if cetak:
        print(f"Populasi {UKURAN_POPULASI} | kromosom {PANJANG_KROMOSOM} bit | "
              f"tournament k={UKURAN_TURNAMEN} | single-point Pc={PC} | "
              f"bit-flip Pm={PM:.4f} | elit {JUMLAH_ELIT} | maks {MAKS_GENERASI} generasi, "
              f"stagnasi {BATAS_STAGNASI}\n")

    populasi = inisialisasi_populasi()
    terbaik, fitness_terbaik = None, float("-inf")
    tanpa_perbaikan = 0

    for generasi in range(1, MAKS_GENERASI + 1):
        fitness = [hitung_fitness(k) for k in populasi]

        i = max(range(UKURAN_POPULASI), key=lambda idx: fitness[idx])
        if fitness[i] > fitness_terbaik:
            fitness_terbaik, terbaik = fitness[i], populasi[i][:]
            tanpa_perbaikan = 0
        else:
            tanpa_perbaikan += 1

        if cetak and (generasi == 1 or generasi % 10 == 0):
            x1, x2 = dekode_kromosom(terbaik)
            print(f"Generasi {generasi:3d} | f = {fungsi_objektif(x1, x2):12.6f} "
                  f"| x1 = {x1:8.4f} | x2 = {x2:8.4f}")

        # Kriteria berhenti: stagnasi atau mencapai generasi maksimum
        if tanpa_perbaikan >= BATAS_STAGNASI:
            if cetak:
                print(f"\nBerhenti di generasi {generasi}: tidak membaik {BATAS_STAGNASI} generasi")
            break
        if generasi < MAKS_GENERASI:
            populasi = ganti_generasi(populasi, fitness)

    x1, x2 = dekode_kromosom(terbaik)
    if cetak:
        kode = "".join(map(str, terbaik))
        print(f"\nKromosom terbaik : {kode} (x1: {kode[:BIT_PER_VARIABEL]}, x2: {kode[BIT_PER_VARIABEL:]})")
        print(f"x1 = {x1:.6f}, x2 = {x2:.6f}, f(x1, x2) = {fungsi_objektif(x1, x2):.6f}")
    return x1, x2, fungsi_objektif(x1, x2)


def baca_input(label, default, tipe):
    """Minta input user; Enter kosong atau input tidak valid = pakai nilai default."""
    teks = input(f"{label} [default {default}]: ").strip()
    if teks == "":
        return default
    try:
        return tipe(teks)
    except ValueError:
        print(f"  Input tidak valid, memakai default {default}")
        return default


def atur_parameter_dari_user():
    """Input manual parameter GA dari user."""
    global UKURAN_POPULASI, PC, PM, MAKS_GENERASI, SEED
    print("=== Input parameter GA (tekan Enter untuk default) ===")
    UKURAN_POPULASI = max(UKURAN_TURNAMEN, baca_input("Ukuran populasi", UKURAN_POPULASI, int))
    PC = min(1.0, max(0.0, baca_input("Probabilitas crossover Pc (0-1)", PC, float)))
    PM = min(1.0, max(0.0, baca_input("Probabilitas mutasi Pm (0-1)", round(PM, 4), float)))
    MAKS_GENERASI = max(1, baca_input("Jumlah generasi maksimum", MAKS_GENERASI, int))
    SEED = baca_input("Seed random (kosongkan = acak)", None, int)
    print()


if __name__ == "__main__":
    atur_parameter_dari_user()
    jalankan_ga()

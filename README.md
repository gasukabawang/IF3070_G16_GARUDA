# PT Intim Line Cargo — 2D Deck Optimization Visualizer
### Tugas Besar IF3070 Dasar Inteligensi Artifisial (Artificial Intelligence)

Aplikasi visualizer interaktif berbasis web untuk memecahkan masalah optimasi penataan muatan kendaraan/barang 2D di atas geladak kapal kargo PT Intim Line Cargo. Sistem ini menggunakan berbagai algoritma *Local Search* dan *Genetic Algorithm* untuk mencari konfigurasi pemuatan terbaik dengan mempertimbangkan batas muatan geladak, dimensi kapal, orientasi barang, dan batasan tanpa tumpang tindih (*zero collision/overlap*).

---

## Kelompok G16 — GARUDA

| No | Nama Lengkap | NIM | Pengerjaan / Kontribusi |
|:--:|:---|:---:|:---|
| 1 | Salma Az Zahra | 18224015 | <ul><li>Mengerjakan Random Restart & Stochastic Hill-Climbing</li><li>Melakukan Testing & Pengujian Algoritma</li><li>Pengerjaan Laporan</li></ul> |
| 2 | Belva Charissa Putri Have | 18224051 | <ul><li>Mengerjakan Hill-Climbing with Sideways Move & Simulated Annealing</li><li>Melakukan Testing & Pengujian Algoritma</li><li>Pengerjaan Laporan</li></ul> |
| 3 | Sherry Eunike | 18224075 | <ul><li>Mengerjakan Steepest Ascent Hill-Climbing & Genetic Algorithm</li><li>Mengerjakan Bonus Fungsi Objektif</li><li>Melakukan Testing & Pengujian Algoritma</li><li>Pengerjaan Laporan</li></ul> |

---

## Panduan Instalasi & Menjalankan Program

### 1. Prasyarat Sistem
- Python 3.10 atau versi yang lebih baru.
- Web browser modern (Google Chrome, Microsoft Edge, Mozilla Firefox, dll).

### 2. Instalasi Dependensi
Buka terminal pada direktori root proyek, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Jalankan server aplikasi Flask:
```bash
python src/app.py
```

### 4. Mengakses Visualizer
Buka browser dan kunjungi alamat lokal:
```
http://127.0.0.1:5000/
```

---

## Algoritma yang Diimplementasikan

Sistem ini mengimplementasikan 6 algoritma pencarian lokal heuristik:

1. **Steepest Ascent Hill-Climbing**
   - Mengevaluasi seluruh suksesor/tetangga valid dari state saat ini.
   - Bergerak secara deterministik ke tetangga yang memberikan peningkatan nilai fungsi objektif terbesar.
   - Berhenti ketika tidak ada lagi tetangga yang memiliki nilai lebih tinggi dari state saat ini (*local optimum*).

2. **Stochastic Hill-Climbing**
   - Mengevaluasi seluruh tetangga valid yang memberikan peningkatan objektif (*better neighbors*).
   - Memilih salah satu tetangga yang lebih baik secara acak, memberikan variasi jalur pencarian dan mengurangi ketergantungan deterministik murni.

3. **Hill-Climbing with Sideways Move**
   - Mengizinkan pencarian melangkah ke tetangga dengan nilai objektif sama (*flat moves/plateau*) hingga batas tertentu (`max_sideways_moves`).
   - Membantu algoritma keluar dari dataran semu (*shoulder/plateau*) sebelum menyatakan kondisi berhenti pada *local optimum*.

4. **Random Restart Hill-Climbing**
   - Melakukan pendakian bukit berulang kali sebanyak `max_restarts`.
   - Ketika terjebak pada *local optimum*, algoritma membangkitkan state awal acak baru yang valid dan memulai pendakian baru.
   - Menyimpan dan menampilkan solusi terbaik global (*overall best*) yang ditemukan dari seluruh rangkaian restart.

5. **Simulated Annealing**
   - Mengadopsi prinsip pendinginan termodinamika logam.
   - Selalu menerima langkah yang meningkatkan nilai objektif ($\Delta E > 0$).
   - Jika tetangga bernilai lebih buruk ($\Delta E \le 0$), perpindahan tetap dapat diterima dengan probabilitas Boltzmann:
     $$P = e^{\Delta E / T}$$
   - Suhu $T$ diturunkan secara bertahap pada setiap langkah dengan laju pendinginan $\alpha$ ($T \leftarrow T \times \alpha$).
   - **Fitur Khusus SA**:
     - Plot kurva probabilitas penerimaan $e^{\Delta E / T}$ terhadap nomor iterasi.
     - Penghitung dan rasio frekuensi terjebak (*stuck*) pada kondisi optimum lokal.

6. **Genetic Algorithm**
   - Pendekatan berbasis populasi dan evolusi biologis:
     - **Inisialisasi**: Membangkitkan populasi acak baru berukuran $k$ pada Generasi 0 (dapat dijalankan berulang kali).
     - **Seleksi**: *Roulette Wheel Selection* berbasis proporsi nilai kebugaran (*fitness*).
     - **Crossover**: *One-Point Crossover* yang dilengkapi mekanisme perbaikan (*repair operator*) agar anak yang dihasilkan selalu memenuhi batasan geladak kapal.
     - **Mutasi**: Mutasi acak berupa pergeseran koordinat, rotasi orientasi, atau perubahan status muat kendaraan berdasarkan `mutation_rate`.
     - **Elitisme**: Individu dengan kebugaran tertinggi selalu dipertahankan langsung ke generasi berikutnya.

---

## Fungsi Objektif (Objective Functions)

Sistem menyediakan 3 pilihan fungsi objektif tanpa mengubah variabel dasar:

1. **Total Shipping Fee (Default)**
   - Memaksimalkan total ongkos kirim seluruh kendaraan yang berhasil dimuat di atas geladak kapal.
   $$f(\text{state}) = \sum_{v \in \text{loaded}} \text{fee}_v$$

2. **Urgency Priority**
   - Memprioritaskan muatan yang memiliki batas sisa hari tiba (`eta`) paling mendesak/kecil.
   $$f(\text{state}) = \sum_{v \in \text{loaded}} \frac{\text{fee}_v}{\max(1, \text{eta}_v)}$$

3. **Space & Capacity Utilization**
   - Menggabungkan perolehan ongkos kirim dengan efisiensi penggunaan luas area geladak ($w \times l$) dan utilisasi kapasitas berat kapal ($MaxCapacity$).

---

## Fitur Visualizer

- **Dual Execution Mode**:
  - *Fast Mode*: Eksekusi langsung hingga kondisi terminasi tercapai dengan visualisasi grafik lengkap.
  - *Animated Mode*: Animasi visual interaktif langkah demi langkah dengan deteksi kilatan tabrakan (*constraint clash flash*).
- **Single Step & Reset State**: Kemampuan melangkah per iterasi secara manual dengan penguncian tombol otomatis saat mencapai *local optimum* (kecuali Genetic Algorithm).
- **Export & Log Riwayat**: Menyimpan hasil eksekusi lengkap ke dalam folder `output/` dalam format `.json` dan `.txt`, serta tabel riwayat interaktif yang dapat dibersihkan secara instan bersamaan dengan penghapusan file output.

---

## Pengaturan & Parameter Algoritma

| Parameter | Keterangan | Rentang Default |
|:---|:---|:---|
| `max_iterations` | Batas maksimum iterasi untuk HC & SA | 10 – 5000 (Default: 500) |
| `max_sideways_moves` | Batas langkah menyamping berturut-turut pada Sideways HC | 1 – 1000 (Default: 100) |
| `max_restarts` | Jumlah iterasi restart acak pada Random Restart HC | 1 – 50 (Default: 5) |
| `initial_temperature` | Suhu awal ($T_0$) pada Simulated Annealing | 10 – 50000 (Default: 1000) |
| `cooling_rate` | Laju pendinginan ($\alpha$) pada Simulated Annealing | 0.50 – 0.999 (Default: 0.95) |
| `population_size` | Ukuran populasi individu ($k$) pada Genetic Algorithm | 4 – 200 (Default: 20) |
| `max_generations` | Jumlah generasi pada Genetic Algorithm | 5 – 500 (Default: 40) |
| `mutation_rate` | Peluang terjadinya mutasi gen pada anak | 0.01 – 1.00 (Default: 0.15) |
# Pengenalan Sampul Buku (Cover Book Recognition)

Repositori ini berisi kode untuk proyek klasifikasi dan pengenalan sampul buku. Sistem ini mampu mengidentifikasi kategori atau judul buku berdasarkan gambar sampulnya. Proyek ini juga dilengkapi dengan antarmuka web sederhana untuk demonstrasi.

## 📝 Deskripsi Proyek

Tujuan utama dari proyek ini adalah membangun model deep learning yang dapat secara akurat mengklasifikasikan buku ke dalam kategori tertentu (misalnya, Fiksi, Non-Fiksi, Sains, Sejarah, dll.) hanya dengan menganalisis gambar sampulnya. Proyek ini juga mencakup komponen OCR (Optical Character Recognition) untuk mengekstrak teks dari sampul sebagai fitur tambahan.

## ✨ Fitur Utama

- **Klasifikasi Gambar**: Model machine learning untuk mengenali dan mengklasifikasikan sampul buku.
- **Ekstraksi Teks (OCR)**: Kemampuan membaca teks (judul, penulis) langsung dari gambar sampul.
- **Antarmuka Web**: Aplikasi web berbasis Flask/FastAPI untuk mengunggah gambar dan melihat hasil prediksi secara real-time.
- **Manajemen Model**: Skrip untuk melatih, menyimpan, memuat, dan mengevaluasi performa model.
- **Tuning Hyperparameter**: Skrip untuk mencari parameter terbaik guna meningkatkan akurasi model.

## 📂 Struktur Direktori & File

```
COVER-BOOK-RECOGNITION/
│
├── saved_models/             # Direktori untuk menyimpan model yang telah dilatih
├── __pycache__/              # Direktori cache Python
│
├── app.py                    # File utama untuk menjalankan aplikasi web (Flask/FastAPI)
├── server.py                 # Skrip server alternatif atau pendukung untuk app.py
│
├── save_model.py             # Skrip untuk melatih dan menyimpan model baru
├── load_model.py             # Skrip atau modul untuk memuat model yang tersimpan
├── evaluate_model.py         # Skrip untuk mengukur performa model dengan data uji
├── new_evaluate_classification.py # Versi baru/alternatif dari skrip evaluasi
├── tune_model.py             # Skrip untuk melakukan tuning hyperparameter model
│
├── ocrgacor.py               # Modul khusus untuk melakukan OCR pada gambar sampul
│
├── book_categories.json      # File JSON berisi daftar kategori buku
├── tuning_results.json       # File untuk menyimpan hasil dari proses tuning
│
├── index.html                # Template HTML untuk antarmuka web
├── requirements.txt          # Daftar pustaka Python yang dibutuhkan
└── README.md                 # Dokumentasi proyek
```

## ⚙️ Instalasi

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di lingkungan lokal Anda:

1. **Clone Repositori**
   ```bash
   git clone https://github.com/Sinduaditya/Assignment-PP.git
   cd COVER-BOOK-RECOGNITION
   git checkout tugas-kelompok
   ```

2. **Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   ```

3. **Instal Dependensi**
   Pastikan semua pustaka yang dibutuhkan terinstal:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Penggunaan

Proyek ini memiliki beberapa komponen yang dapat dijalankan secara terpisah.

### 1. Menjalankan Aplikasi Web

Untuk memulai server dan mencoba aplikasi melalui browser:

```bash
python app.py
```

Setelah server berjalan, buka browser Anda dan akses `http://127.0.0.1:5000` (atau port lain yang Anda konfigurasikan).

### 2. Melatih Model Baru

Untuk melatih ulang model dengan dataset Anda:

```bash
python save_model.py
```

_Catatan: Pastikan dataset telah disiapkan dan path-nya dikonfigurasi di dalam skrip._

### 3. Mengevaluasi Model

Untuk menguji akurasi model terhadap dataset evaluasi:

```bash
python evaluate_model.py
```

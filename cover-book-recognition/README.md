# Pengenalan Sampul Buku (Cover Book Recognition)

Repositori ini berisi kode untuk proyek klasifikasi dan pengenalan sampul buku. Sistem ini mampu mengidentifikasi kategori atau judul buku berdasarkan gambar sampulnya. Proyek ini juga dilengkapi dengan antarmuka web sederhana untuk demonstrasi.

## 📝 Deskripsi Proyek

Tujuan utama dari proyek ini adalah untuk membangun model deep learning yang dapat secara akurat mengklasifikasikan buku ke dalam kategori yang telah ditentukan (misalnya, Fiksi, Non-Fiksi, Sains, Sejarah, dll.) hanya dengan menganalisis gambar sampulnya. Proyek ini juga mencakup komponen OCR (Optical Character Recognition) untuk mengekstrak teks dari sampul sebagai fitur tambahan.

## ✨ Fitur Utama

- **Klasifikasi Gambar**: Model machine learning untuk mengenali dan mengklasifikasikan sampul buku.
- **Ekstraksi Teks (OCR)**: Kemampuan untuk membaca teks (judul, penulis) langsung dari gambar sampul.
- **Antarmuka Web**: Aplikasi web berbasis Flask/FastAPI untuk mengunggah gambar dan melihat hasil prediksi secara real-time.
- **Manajemen Model**: Skrip untuk melatih, menyimpan, memuat, dan mengevaluasi performa model.
- **Tuning Hyperparameter**: Skrip untuk mencari parameter terbaik guna meningkatkan akurasi model.

## 📂 Struktur Direktori & File

COVER-BOOK-RECOGNITION/
│
├── saved_models/ # Direktori untuk menyimpan model yang telah dilatih
├── **pycache**/ # Direktori cache Python
│
├── app.py # File utama untuk menjalankan aplikasi web (kemungkinan Flask/FastAPI)
├── server.py # Skrip server alternatif atau pendukung untuk app.py
│
├── save_model.py # Skrip untuk melatih dan menyimpan model baru
├── load_model.py # Skrip atau modul untuk memuat model yang tersimpan
├── evaluate_model.py # Skrip untuk mengukur performa model dengan data uji
├── new_evaluate_classification.py # Versi baru/alternatif dari skrip evaluasi
├── tune_model.py # Skrip untuk melakukan tuning hyperparameter model
│
├── ocrgacor.py # Modul khusus untuk melakukan OCR pada gambar sampul
│
├── book_categories.json # File JSON yang berisi daftar kategori buku
├── tuning_results.json # File untuk menyimpan hasil dari proses tuning
│
├── index.html # Template HTML untuk antarmuka web
├── requirements.txt # Daftar pustaka (library) Python yang dibutuhkan
└── README.md # File ini (dokumentasi proyek)

````

## ⚙️ Instalasi

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/JovanSantosa/cover-book-recognition.git]
    cd COVER-BOOK-RECOGNITION
    ```

2.  **Buat dan Aktifkan Lingkungan Virtual (Direkomendasikan)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
    ```

3.  **Instal Dependensi**
    Pastikan semua pustaka yang dibutuhkan terinstal dengan menjalankan perintah berikut:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Penggunaan

Proyek ini memiliki beberapa komponen yang dapat dijalankan secara terpisah.

### 1. Menjalankan Aplikasi Web

Untuk memulai server dan mencoba aplikasi melalui browser:

```bash
python app.py
````

Setelah server berjalan, buka browser Anda dan akses alamat `http://127.0.0.1:5000` (atau port lain yang Anda konfigurasikan).

### 2. Melatih Model Baru

Untuk melatih ulang model dengan dataset Anda:

```bash
python save_model.py
```

_Catatan: Pastikan Anda telah menyiapkan dataset dan mengonfigurasi path-nya di dalam skrip._

### 3. Mengevaluasi Model

Untuk menguji akurasi model yang ada terhadap dataset evaluasi:

```bash
python evaluate_model.py
```

---

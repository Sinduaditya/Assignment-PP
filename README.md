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

# EL Class Pattern Recognition Assignment Repo

## Petunjuk Bagi Mahasiswa

Untuk mengerjakan tugas, ikuti langkah-langkah berikut:

1. **Fork Repository Ini**  
   - Klik tombol **Fork** di sudut kanan atas halaman repository ini untuk membuat salinan repository ke akun GitHub Anda.

2. **Clone Fork Repository Anda ke Local**  
   - Buka terminal dan jalankan perintah berikut (ganti `<username>` dengan username GitHub Anda):
     ```bash
     git clone https://github.com/<username>/Assignment.git
     ```
   
3. **Buat Branch di Fork Anda**  
   - Masuk ke direktori repository yang telah di-clone:
     ```bash
     cd Assignment
     ```
   - Buat branch baru dengan nama `Assignment-n`, di mana `n` adalah nomor tugas. Contoh: untuk Tugas 1, gunakan:
     ```bash
     git checkout -b Assignment-1
     ```

4. **Kerjakan Tugas Anda**  
   - Tambahkan file-file hasil pengerjaan tugas Anda, baik file **.ipynb** maupun **.py** ke dalam branch yang telah dibuat.

5. **Push File ke Fork Repository Anda**  
   - Setelah selesai mengerjakan, jalankan perintah berikut untuk mengirim perubahan ke GitHub:
     ```bash
     git add .
     git commit -m "Tugas 1 - Penugasan EL Class Pattern Recognition"
     git push origin Assignment-1
     ```

6. **Lakukan Pull Request ke Repository Utama (Upstream)**  
   - Buka repository fork Anda di GitHub.
   - Klik tombol **Compare & pull request** pada branch `Assignment-1`.
   - Buat pull request ke repository utama.
   - Tunggu hingga seluruh pemeriksaan (checks) selesai sebelum pull request Anda di-merge.

7. **Selesai**  
   - Setelah pull request diterima dan semua pemeriksaan selesai, tugas Anda dianggap selesai.

Selamat mengerjakan!

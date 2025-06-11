import json
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score, make_scorer
import numpy as np
from collections import Counter

# --- Data Konfigurasi & Sampel ---
book_categories_data = [
    {"category_name":"001 - Ilmu Pengetahuan Umum","rack_id":"RK001","keywords":["ensiklopedia","umum","fakta","pengetahuan","referensi","sejarah","budaya","dunia","modern","dasar","ielts","penemuan"]},
    {"category_name":"003 - Sistem","rack_id":"RK003","keywords":["sistem","arsitektur","desain","analisis","manajemen","informasi","kompleks","struktur","pemodelan","implementasi","dairy","teacher","bisnis"]},
    {"category_name":"004 - Pengolahan Data","rack_id":"RK004","keywords":["data","pengolahan","analisis","big data","database","statistik","visualisasi","miner","machine learning","algoritma","prediksi","kecerdasan buatan"]},
    {"category_name":"005 - Pemrograman Komputer","rack_id":"RK005","keywords":["pemrograman","komputer","python","java","c++","web","aplikasi","algoritma","coding","software","pengembangan","android","ios","jaringan"]},
    {"category_name":"160 - Logika","rack_id":"RK160","keywords":["logika","filsafat","penalaran","kritis","matematika","argumen","deduktif","induktif","simbolik","pemikiran","kebenaran"]},
    {"category_name":"512 - Aljabar","rack_id":"RK512","keywords":["aljabar","persamaan","variabel","matriks","vektor","polinomial","linear","abstrak","grup","cincin","matematika dasar"]},
    {"category_name":"519 - Probabilitas dan Matematika Terapan","rack_id":"RK519","keywords":["probabilitas","statistik","terapan","model","data","inferensi","stokastik","simulasi","riset","operasi","prediksi","random","kuantitatif"]},
    {"category_name":"531 - Mekanika Klasik","rack_id":"RK531","keywords":["mekanika","klasik","newton","gerak","gaya","energi","momentum","sistem","partikel","osilasi","fisika dasar","hukum"]},
    {"category_name":"539 - Fisika Modern","rack_id":"RK539","keywords":["fisika","materi","energi","gerak","listrik","mekanika","kuantum","relativitas","termodinamika","gelombang","nuklir","semikonduktor"]},
    {"category_name":"547 - Kimia Organik","rack_id":"RK547","keywords":["kimia","unsur","senyawa","reaksi","atom","molekul","ikatan","organik","anorganik","laboratorium","sintesis","biokimia"]}
]

# Dataset sampel (DIREPLIKASI UNTUK MENDAPATKAN LEBIH BANYAK DATA)
# Ini hanya untuk tujuan demonstrasi agar ML model bisa belajar.
# Di aplikasi nyata, Anda harus memiliki dataset yang lebih besar dan beragam.
base_sample_data = [
    ("ensiklopedia lengkap dunia modern", "001 - Ilmu Pengetahuan Umum"),
    ("belajar ielts dari dasar", "001 - Ilmu Pengetahuan Umum"),
    ("desain dan analisis sistem informasi", "003 - Sistem"),
    ("pemodelan arsitektur sistem kompleks", "003 - Sistem"),
    ("pengantar machine learning dan big data", "004 - Pengolahan Data"),
    ("analisis statistik dengan python", "004 - Pengolahan Data"),
    ("manajemen basis data relasional", "004 - Pengolahan Data"),
    ("dasar pemrograman java untuk pemula", "005 - Pemrograman Komputer"),
    ("membuat aplikasi web dengan python", "005 - Pemrograman Komputer"),
    ("rekayasa perangkat lunak dan manajemen proyek", "005 - Pemrograman Komputer"),
    ("pengantar filsafat dan logika", "160 - Logika"),
    ("dasar-dasar logika matematika", "160 - Logika"),
    ("fisika kuantum untuk semua", "539 - Fisika Modern"),
    ("teori relativitas dan fisika modern", "539 - Fisika Modern"),
    ("pengantar mekanika klasik newton", "531 - Mekanika Klasik"),
    ("hukum gerak dan energi mekanika", "531 - Mekanika Klasik"),
    ("studi tentang aljabar linear", "512 - Aljabar"),
    ("teori grup dalam aljabar", "512 - Aljabar"),
    ("teori probabilitas dan statistika terapan", "519 - Probabilitas dan Matematika Terapan"),
    ("model stokastik dan probabilitas", "519 - Probabilitas dan Matematika Terapan"),
    ("kimia organik sintesis dan reaksi", "547 - Kimia Organik"),
    ("dasar-dasar kimia organik", "547 - Kimia Organik"),
]

# Mereplikasi data untuk mensimulasikan dataset yang lebih besar
sample_data = base_sample_data * 10 # Menggandakan data 10 kali
# Anda bisa lebih cerdas dalam mereplikasi, misal menambahkan sedikit noise atau variasi.
# Untuk tujuan ini, replikasi sederhana sudah cukup untuk menunjukkan perbedaan.


titles = [d[0] for d in sample_data]
labels = [d[1] for d in sample_data]

print(f"Total sampel data setelah replikasi: {len(titles)}")
print(f"Jumlah kategori unik: {len(set(labels))}")

# --- Langkah 1: Membagi Data ---
# Gunakan test_size yang lebih kecil untuk memastikan ada cukup data pelatihan
# test_size=0.3 artinya 30% untuk test, 70% untuk train
# stratify=labels memastikan distribusi kelas yang serupa di train dan test set
X_train, X_test, y_train, y_test = train_test_split(
    titles, labels, test_size=0.3, random_state=42, stratify=labels
)

print("\nDistribusi kelas di y_train:")
print(Counter(y_train))
print("\nDistribusi kelas di y_test:")
print(Counter(y_test))

# --- Langkah 2: Evaluasi Baseline Model (Keyword Matching) ---

def baseline_categorize_text(text, categories):
    text = text.lower()
    best_match = None
    highest_score = -1 # Mulai dari -1 agar 0 score pun bisa ditangkap
    
    # Kumpulkan semua category_name yang ada
    all_category_names = {cat['category_name'] for cat in categories}

    for category in categories:
        score = 0
        for keyword in category['keywords']:
            if keyword.lower() in text:
                score += 1
        
        # Jika skor lebih tinggi, update
        if score > highest_score:
            highest_score = score
            best_match = category['category_name']
        # Jika skor sama dan sudah ada best_match, abaikan
        # Jika skor sama dan belum ada best_match (misal semua 0), ambil yang pertama
        elif score == highest_score and best_match is None: # Ini untuk kasus score 0
            best_match = category['category_name'] 
            
    # Jika tidak ada kecocokan keyword sama sekali (score tetap -1 atau 0)
    # Ini bisa terjadi jika teks tidak mengandung keyword apapun
    return best_match if best_match else "Unknown"

baseline_predictions = [baseline_categorize_text(text, book_categories_data) for text in X_test]

# --- Langkah 3: Evaluasi Best Model (Machine Learning: TF-IDF + SVM) ---

ml_model = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear', random_state=42, C=1.0) # C=1.0 adalah nilai default, bisa disetel
)
ml_model.fit(X_train, y_train) # Latih model dengan data pelatihan
ml_predictions = ml_model.predict(X_test) # Prediksi pada data uji

# --- Langkah 4: Menampilkan Hasil Evaluasi Train/Test Split ---

print("\n" + "=" * 70)
print("          LAPORAN EVALUASI MODEL (TRAIN/TEST SPLIT)")
print("=" * 70)

# Pastikan semua label yang mungkin muncul di laporan ada
all_labels = sorted(list(set(labels) | set(baseline_predictions) | set(ml_predictions)))
if "Unknown" in baseline_predictions and "Unknown" not in all_labels:
    all_labels.append("Unknown") # Tambahkan 'Unknown' jika baseline memprediksinya

print("\n--- Baseline Model (Keyword Matching) ---")
report_baseline = classification_report(
    y_test,
    baseline_predictions,
    labels=all_labels,
    zero_division=0,
    digits=4 # Menampilkan 4 angka desimal
)
print(report_baseline)

print("\n--- Best Model (TF-IDF + SVM) ---")
report_ml = classification_report(
    y_test,
    ml_predictions,
    labels=all_labels,
    zero_division=0,
    digits=4 # Menampilkan 4 angka desimal
)
print(report_ml)

print("=" * 70)
print("Analisis Sederhana (Train/Test Split):")
print(f"Data Pelatihan: {len(y_train)} sampel")
print(f"Data Uji: {len(y_test)} sampel")
print("Model Baseline mengandalkan pencocokan kata kunci statis.")
print("Model Machine Learning belajar pola dari data,")
print("memungkinkannya untuk menggeneralisasi ke judul baru.")
print("Perhatikan akurasi yang lebih baik pada model ML berkat data yang lebih banyak.")
print("=" * 70)


# --- Langkah 5: Evaluasi Menggunakan Cross-Validation (Disarankan untuk Dataset Kecil) ---

print("\n\n" + "=" * 70)
print("            EVALUASI MODEL MENGGUNAKAN CROSS-VALIDATION")
print("=" * 70)

# Definisikan scorer untuk setiap metrik yang Anda inginkan
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0) # Menambahkan recall
}

# Model untuk Cross-Validation (bisa sama dengan model yang di atas)
ml_model_cv = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear', random_state=42, C=1.0)
)

# Lakukan K-Fold Cross-Validation (misalnya, 5-fold)
# Ini akan membagi dataset menjadi 5 bagian, melatih pada 4 bagian, dan menguji pada 1 bagian,
# mengulang 5 kali dengan bagian uji yang berbeda.
n_splits_cv = 5 # Jumlah fold untuk CV
print(f"\nMelakukan {n_splits_cv}-Fold Cross-Validation untuk Best Model (TF-IDF + SVM)...")
cv_results_ml = cross_validate(ml_model_cv, titles, labels, cv=n_splits_cv, scoring=scoring, return_train_score=False)

print("\n--- Hasil Cross-Validation untuk Best Model (TF-IDF + SVM) ---")
print(f"Rata-rata Akurasi: {cv_results_ml['test_accuracy'].mean():.4f} (+/- {cv_results_ml['test_accuracy'].std():.4f})")
print(f"Rata-rata Presisi (Weighted): {cv_results_ml['test_precision_weighted'].mean():.4f} (+/- {cv_results_ml['test_precision_weighted'].std():.4f})")
print(f"Rata-rata Recall (Weighted): {cv_results_ml['test_recall_weighted'].mean():.4f} (+/- {cv_results_ml['test_recall_weighted'].std():.4f})")
print(f"Rata-rata F1-Score (Weighted): {cv_results_ml['test_f1_weighted'].mean():.4f} (+/- {cv_results_ml['test_f1_weighted'].std():.4f})")

print("\n--- Melakukan Cross-Validation untuk Baseline Model (Keyword Matching) ---")
# Untuk baseline model, kita harus membuat loop manual untuk CV
baseline_accuracies = []
baseline_precisions = []
baseline_f1_scores = []
baseline_recalls = []

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=42)

for train_index, test_index in skf.split(titles, labels):
    X_train_fold = [titles[i] for i in train_index]
    y_train_fold = [labels[i] for i in train_index]
    X_test_fold = [titles[i] for i in test_index]
    y_test_fold = [labels[i] for i in test_index]

    # Prediksi untuk baseline model pada fold saat ini
    fold_baseline_predictions = [baseline_categorize_text(text, book_categories_data) for text in X_test_fold]
    
    # Pastikan 'Unknown' ada di label jika diperlukan untuk perhitungan metrik
    current_all_labels = sorted(list(set(y_test_fold) | set(fold_baseline_predictions)))
    if "Unknown" in fold_baseline_predictions and "Unknown" not in current_all_labels:
        current_all_labels.append("Unknown")

    baseline_accuracies.append(accuracy_score(y_test_fold, fold_baseline_predictions))
    baseline_precisions.append(precision_score(y_test_fold, fold_baseline_predictions, average='weighted', zero_division=0, labels=current_all_labels))
    baseline_f1_scores.append(f1_score(y_test_fold, fold_baseline_predictions, average='weighted', zero_division=0, labels=current_all_labels))
    baseline_recalls.append(recall_score(y_test_fold, fold_baseline_predictions, average='weighted', zero_division=0, labels=current_all_labels))

print("\n--- Hasil Cross-Validation untuk Baseline Model ---")
print(f"Rata-rata Akurasi: {np.mean(baseline_accuracies):.4f} (+/- {np.std(baseline_accuracies):.4f})")
print(f"Rata-rata Presisi (Weighted): {np.mean(baseline_precisions):.4f} (+/- {np.std(baseline_precisions):.4f})")
print(f"Rata-rata Recall (Weighted): {np.mean(baseline_recalls):.4f} (+/- {np.std(baseline_recalls):.4f})")
print(f"Rata-rata F1-Score (Weighted): {np.mean(baseline_f1_scores):.4f} (+/- {np.std(baseline_f1_scores):.4f})")


print("\n" + "=" * 70)
print("Kesimpulan Akhir:")
print("Metrik Cross-Validation memberikan gambaran kinerja model yang lebih stabil")
print("karena model dievaluasi pada beberapa subset data yang berbeda.")
print("Biasanya, model Machine Learning akan mengungguli model berbasis aturan")
print("jika data pelatihan cukup representatif.")
print("Pastikan untuk mengganti data dummy dengan data nyata Anda.")
print("=" * 70)
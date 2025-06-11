import math

# Data frekuensi kata
terms = [
    "sentiment", "prediction", "streamlit", "model", "grab", "reviews",
    "svm", "linear", "feature", "visualization", "user", "evaluation", "accuracy",
    "classification", "data", "balancing", "techniques", "performance",
    "SMOTE", "obesity", "health", "minority", "study",
    "analysis", "multi-domain", "CNN-LSTM", "LSTM", "CNN", "domain",
    "IKN", "project", "public", "opinions", "characteristics"
]

doc1 = [6, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

doc2 = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2,
        3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

doc3 = [4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
        4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]

documents = [doc1, doc2, doc3]
N = len(documents)

# Hitung TF
def compute_tf(doc):
    total_terms = sum(doc)
    return [count / total_terms if total_terms != 0 else 0 for count in doc]

tf_docs = [compute_tf(doc) for doc in documents]

# Hitung DF (jumlah dokumen yang mengandung term)
df = []
for i in range(len(terms)):
    count = sum(1 for doc in documents if doc[i] > 0)
    df.append(count)

# Hitung IDF
idf = [math.log((N + 1) / (df_i + 1)) + 1 for df_i in df]  # +1 smoothing

# Hitung TF-IDF
tfidf_docs = []
for tf in tf_docs:
    tfidf = [tf[i] * idf[i] for i in range(len(terms))]
    tfidf_docs.append(tfidf)

# Fungsi cosine similarity
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

# Hitung cosine similarity antar pasangan dokumen
sim_1_2 = cosine_similarity(tfidf_docs[0], tfidf_docs[1])
sim_1_3 = cosine_similarity(tfidf_docs[0], tfidf_docs[2])
sim_2_3 = cosine_similarity(tfidf_docs[1], tfidf_docs[2])

# Tampilkan hasil
print("Cosine Similarity:")
print(f"Doc 1 vs Doc 2: {sim_1_2:.4f}")
print(f"Doc 1 vs Doc 3: {sim_1_3:.4f}")
print(f"Doc 2 vs Doc 3: {sim_2_3:.4f}")

matrix = [
    [1.00, sim_1_2, sim_1_3],
    [sim_1_2, 1.00, sim_2_3],
    [sim_1_3, sim_2_3, 1.00]
]

# Header dan index
headers = ["Doc 1", "Doc 2", "Doc 3"]
print("Cosine Similarity Matrix:")
print("        " + "  ".join(f"{h:>7}" for h in headers))
for i, row in enumerate(matrix):
    row_str = "  ".join(f"{val:7.4f}" for val in row)
    print(f"{headers[i]:<7} {row_str}")

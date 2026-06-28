# Proyek Machine Learning: Clustering & Klasifikasi Transaksi Keuangan

**Submission Akhir — Belajar Machine Learning untuk Pemula (BMLP) | Dicoding**

> Dataset: Transaksi keuangan untuk eksplorasi **deteksi penipuan (fraud detection)**, mencakup pola perilaku transaksi dan aktivitas nasabah.

---

## Struktur File

| File | Deskripsi |
|------|-----------|
| `[Clustering]_Submission_Akhir_BMLP_Kurnia_Andre_Febrian.ipynb` | Notebook utama clustering |
| `[Klasifikasi]_Submission_Akhir_BMLP_Kurnia_Andre_Febrian.ipynb` | Notebook utama klasifikasi |
| `data_clustering.csv` | Dataset hasil clustering (scaled, dengan kolom `Target`) |
| `data_clustering_inverse.csv` | Dataset hasil clustering (nilai asli/inverse) |
| `model_clustering.h5` | Model K-Means yang telah dilatih |
| `PCA_model_clustering.h5` | Model PCA untuk visualisasi clustering |
| `decision_tree_model.h5` | Model Decision Tree yang telah dilatih |
| `explore_RandomForest_classification.h5` | Model Random Forest hasil eksplorasi |
| `tuning_classification.h5` | Model klasifikasi hasil hyperparameter tuning |

---

## Notebook 1: Clustering

### Section 1 — Import Library
Mengimpor seluruh pustaka yang dibutuhkan:
- `pandas`, `numpy` — manipulasi data
- `matplotlib`, `seaborn` — visualisasi
- `sklearn` — preprocessing, clustering, evaluasi
- `yellowbrick` — visualisasi Elbow Method
- `joblib` — menyimpan model

### Section 2 — Memuat Dataset & EDA
- Memuat dataset menggunakan `pd.read_csv()`
- Menampilkan 5 baris pertama dengan `head()`
- Mengecek informasi dataset dengan `info()`
- Statistik deskriptif dengan `describe()`
- **(Skilled)** Visualisasi korelasi antar fitur dan histogram distribusi
- **(Advanced)** Boxplot `TransactionAmount` berdasarkan `CustomerOccupation`

### Section 3 — Pembersihan & Pra Pemrosesan Data
- Mengecek missing values dengan `isnull().sum()`
- Mengecek duplikat dengan `duplicated().sum()`
- Menghapus missing values dengan `dropna()`
- Menghapus duplikat dengan `drop_duplicates()`
- Drop kolom tidak relevan: `TransactionID`, `AccountID`, `DeviceID`, `IPAddress`, `MerchantID`, `Date`
- Encoding fitur kategorikal dengan `LabelEncoder`
- **(Skilled)** Handling outlier dengan metode IQR
- **(Skilled)** Feature scaling dengan `StandardScaler`
- **(Advanced)** Binning data numerik

### Section 4 — Membangun Model Clustering
- Memastikan data menggunakan hasil preprocessing
- Menentukan jumlah cluster optimal dengan **Elbow Method** (`KElbowVisualizer`)
- Melatih model **K-Means** (`n_clusters=2`, `random_state=42`)
- Menyimpan model dengan `joblib.dump()`
- **(Skilled)** Menghitung **Silhouette Score**
- **(Skilled)** Visualisasi hasil clustering dengan **PCA** (2 komponen)
- **(Advanced)** Perbandingan model menggunakan PCA

### Section 5 — Interpretasi Cluster
Analisis deskriptif (mean, min, max) per cluster dalam kondisi **scaled**:

| Fitur | Cluster 0 (mean) | Cluster 1 (mean) |
|-------|-----------------|-----------------|
| TransactionAmount | -0.01 | 0.01 |
| CustomerAge | 0.02 | -0.02 |
| TransactionDuration | 0.03 | -0.03 |
| LoginAttempts | 0.00 | 0.00 |
| AccountBalance | 0.01 | -0.01 |

**Cluster 0 — Profesional Dewasa (Charlotte, Debit, Branch, Doctor, Dewasa)**
Nasabah dengan profil stabil, TransactionDuration dan AccountBalance sedikit di atas rata-rata. Tidak ada indikasi aktivitas mencurigakan (LoginAttempts = 0.00 scaled).

**Cluster 1 — Pelajar/Muda (Tucson, Debit, Branch, Student, Muda)**
Nasabah muda dengan TransactionAmount sedikit lebih tinggi namun TransactionDuration dan AccountBalance lebih rendah dari Cluster 0. Pola login juga stabil.

- **(Skilled)** Analisis deskriptif pada data **inverse** (nilai asli):

| Fitur | Cluster 0 | Cluster 1 |
|-------|-----------|-----------|
| TransactionAmount (mean) | 255.55 | 258.15 |
| CustomerAge (mean) | 45.06 thn | 44.33 thn |
| TransactionDuration (mean) | 121.12 dtk | 117.30 dtk |
| LoginAttempts (mean) | 1.00 | 1.00 |
| AccountBalance (mean) | 5,142.17 | 5,058.81 |

### Section 6 — Mengeksport Data
- Menyimpan hasil clustering ke `data_clustering.csv` (kolom cluster bernama `Target`)
- **(Skilled)** Menyimpan data inverse ke `data_clustering_inverse.csv`

---

## Notebook 2: Klasifikasi

### Section 1 — Import Library
Mengimpor pustaka untuk klasifikasi: `sklearn`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, dll.

### Section 2 — Memuat Dataset
- Memuat `data_clustering.csv` hasil notebook clustering sebagai input klasifikasi
- Kolom `Target` digunakan sebagai label kelas

### Section 3 — Pembersihan & Pra Pemrosesan Data
- Split data menjadi train dan test menggunakan `train_test_split()`
- Data sudah bersih dari proses clustering sebelumnya

### Section 4 — Membangun Model Klasifikasi
- Melatih model **Decision Tree** sebagai model baseline
- Menyimpan model dengan `joblib.dump()` sebagai `decision_tree_model.h5`
- **(Skilled)** Eksplorasi **Random Forest** (`explore_RandomForest_classification.h5`)
- **(Advanced)** Hyperparameter tuning dengan **GridSearchCV** (`tuning_classification.h5`)

### Section 5 — Evaluasi Model
- Mengevaluasi performa model menggunakan metrik klasifikasi
- Menampilkan hasil evaluasi secara otomatis via `joblib`

---

## Hasil

- **Jumlah Cluster:** 2
- **Algoritma Clustering:** K-Means
- **Silhouette Score:** dihitung otomatis saat notebook dijalankan
- **Algoritma Klasifikasi:** Decision Tree (baseline) + Random Forest + Tuning

---

*Dikerjakan oleh: **Kurnia Andre Febrian***
*Program: Belajar Machine Learning untuk Pemula — Dicoding x Microsoft Elevate 2025*

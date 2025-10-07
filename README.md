#  AI/ML Mini Projects

Proyek ini terdiri dari **2 tugas** yang fokus pada *machine learning pipeline* dari pembangunan model, evaluasi, hingga deployment.

---

##  Project 1: Sentiment Analysis Model

### Deskripsi

* Dataset: **IMDB 50K Movie Reviews** (2 kelas: positive, negative).
* Metode: **TF-IDF Vectorizer + Logistic Regression**.
* Fokus: Preprocessing teks, training model, hyperparameter tuning, dan evaluasi.

### Pembahasan

1. **Preprocessing**

   * Konversi huruf kecil.
   * Hapus HTML tags.
   * Hapus karakter non-alfabet.
   * Representasi teks menggunakan **TF-IDF** (maks 5000 fitur).

2. **Model**

   * Algoritma: **Logistic Regression**.
   * Hyperparameter Tuning manual pada `C` dan `solver`.

3. **Hasil Evaluasi Terbaik**

   ```
   --- Classification Report (Best Model) ---
                 precision    recall  f1-score   support

       negative       0.89      0.88      0.89      5000
       positive       0.88      0.90      0.89      5000

       accuracy                           0.89     10000
      macro avg       0.89      0.89      0.89     10000
   ```

   ✅ Model konsisten dengan akurasi **~89%**, seimbang antara kelas positive & negative.

---

##  Project 2: Image Classification & Deployment

### Deskripsi

* Dataset: **CIFAR-10** (10 kelas gambar, ukuran 32x32).
* Model: **CNN sederhana (Keras/TensorFlow)**.
* Deployment: **Flask API** dengan Docker.

### Pembahasan

1. **Model CNN**

   * 2 convolutional blocks (Conv2D + MaxPooling + Dense).
   * Dense layer untuk klasifikasi 10 kelas.
   * Optimizer: Adam.

2. **Deployment**

   * Model diload di Flask.
   * Endpoint `/predict` menerima file gambar → output class + confidence score.
   * Docker digunakan untuk containerization agar mudah dijalankan di mana saja.

### Cara Menjalankan

#### 1. Training Model

```bash
python train.py
```

Model terlatih akan tersimpan sebagai `cifar10_model.h5`.

#### 2. Menjalankan Flask API

```bash
python app.py --port 8000
```

API aktif di `http://127.0.0.1:8000/predict`.

#### 3. Testing dengan Script

```bash
python test_request.py
```

Contoh output:

```json
{"class": "cat", "confidence": 0.9942}
```

#### 4. Docker Deployment

Build image:

```bash
docker build -t cifar10-app .
```

Run container:

```bash
docker run -p 8000:8000 cifar10-app
```

#### 5. Alternatif Testing dengan `curl`

Jika script Python tidak bisa dijalankan, bisa gunakan `curl` untuk mengirim request:

```bash
curl -X POST -F "cat.png" http://127.0.0.1:8000/predict
```

Output akan berupa JSON dengan **prediksi class + confidence**.

---

## Kesimpulan

* **Project 1** berhasil membangun *baseline text classifier* dengan akurasi 89% menggunakan Logistic Regression.
* **Project 2** tidak hanya membangun CNN untuk image classification, tapi juga mendemonstrasikan *end-to-end deployment* dengan Flask + Docker, serta menyediakan alternatif testing via `curl`.

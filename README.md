---
title: Nutrify AI API
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# 🍽️ Nutrify AI — Indonesian Food Recognition & Nutrition API

## 📌 Deskripsi Proyek
**Nutrify AI API** adalah layanan **FastAPI** yang mengenali makanan tradisional Indonesia dari gambar menggunakan model **CNN (TensorFlow / Keras)**, lalu memberikan informasi nutrisi beserta rekomendasi personalisasi berbasis **Gemini LLM**.

Project ini bagian dari **Capstone Project tim AI Engineer (CC26-PSU017)** — DBS Coding Camp 2026. Model & backend disediakan oleh tim AI Engineer untuk dikonsumsi tim Front-End (React + Express).

---

## 🧠 Fitur Utama
- 🔍 **Food Recognition** — klasifikasi gambar makanan Indonesia dengan model CNN
- 🥗 **Nutrition Lookup** — info nutrisi dari database TKPI (Kemenkes)
- 🧮 **Analisa Gabungan** — `/predict` menerima gambar, input manual, atau keduanya, plus personalisasi berdasarkan penyakit
- 🤖 **Chat & Recommendation** — chatbot & rekomendasi makanan berbasis Gemini LLM
- ⚡ **REST API** — endpoint cepat berbasis FastAPI, siap dikonsumsi frontend

---

## 🗂️ Struktur Proyek
```
nutrify-ai-api/
├── api/
│   └── index.py                    # entry point serverless (Vercel)
├── app/
│   ├── controllers/
│   │   └── chat_controller.py      # orkestrasi alur chat
│   ├── core/
│   │   ├── config.py               # config (pydantic-settings)
│   │   └── logging_config.py       # setup logging
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat_model.py           # model/skema data chat
│   │   └── cnn_layers.py           # custom layer model CNN
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat_route.py           # endpoint /chat
│   │   ├── check_system_api.py     # endpoint / & health check
│   │   ├── food.py                 # endpoint /search-food, /units, /disease
│   │   └── predict.py              # endpoint /predict
│   ├── schemas/                    # request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py         # logika chat (Gemini)
│   │   ├── cnn_services.py         # load & inferensi model CNN
│   │   ├── dependencies.py         # shared dependencies (DI)
│   │   ├── nutrition_service.py    # lookup nutrisi (TKPI)
│   │   └── recommendation_service.py  # rekomendasi personalisasi
│   ├── __init__.py
│   └── main.py                     # entry point FastAPI
├── app.py                          # entry point lokal
├── class_names.json                # daftar label kelas makanan
├── indonesian_food_clean.csv       # database nutrisi (TKPI)
├── nutrify_model.keras             # model CNN (Git LFS)
├── Dockerfile
├── requirements.txt
├── .python-version
├── .gitattributes                  # konfigurasi Git LFS
├── .gitignore
├── .env.example                    # template environment variables
├── .env                            # secrets lokal — JANGAN di-commit
└── README.md
```

> 🔐 File `.env` berisi kredensial (mis. `GEMINI_API_KEY`). Pastikan masuk `.gitignore` dan tidak pernah ikut ter-push. Untuk template-nya, lihat `.env.example`.

---

## 📊 Sumber Data
Database nutrisi: `indonesian_food_clean.csv` — bersumber dari **TKPI (Tabel Komposisi Pangan Indonesia, Kemenkes)**.

| Kolom | Deskripsi |
|-------|-----------|
| `food_name` | Nama makanan |
| `calories` | Energi per 100 g (kkal) |
| `protein` | Protein per 100 g (g) |
| `fat` | Lemak per 100 g (g) |
| `carbohydrates` | Karbohidrat per 100 g (g) |

---

## 🔌 Daftar Endpoint

### System
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | Info dasar API |
| `GET` | `/health-check` | Cek status service (health check) |

### Food Data
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/search-food` | Cari makanan di database nutrisi — query param `?q=nama&limit=5` |
| `GET` | `/units` | Daftar satuan porsi + konversi ke gram |
| `GET` | `/disease` | Daftar penyakit untuk rekomendasi personalisasi |

### Prediction
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/predict` | Analisa makanan — terima gambar, input manual, atau keduanya |

### Chat
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/chat/` | Kirim pesan chat & rekomendasi (Gemini) |
| `GET` | `/chat/history/{conversation_id}` | Ambil riwayat percakapan |

> 📖 Dokumentasi interaktif lengkap tersedia di `/docs` (Swagger UI).

---

## 🎯 Panduan `/predict` (penting buat Front-End)

Endpoint utama untuk analisa makanan. Ada **3 input** dan bisa dipakai fleksibel: **gambar saja**, **manual saja**, atau **keduanya**.

| Param | Tipe | Wajib? | Keterangan |
|-------|------|--------|------------|
| `image` | file | opsional* | Foto makanan (`.jpg`, `.png`, `.webp`, dll). Satu foto per request. |
| `manual_items` | JSON array (string) | opsional* | Daftar makanan yang diinput manual. |
| `disease` | string | opsional | Salah satu: `obesitas`, `diabetes`, `hipertensi`, `asam_urat`, `kolesterol`. |

> \*Minimal salah satu dari `image` atau `manual_items` harus diisi.

**Format `manual_items`** — wajib JSON array:
```json
[
  { "food_name": "nasi putih", "quantity": 2, "unit": "porsi" },
  { "food_name": "Ayam Bakar", "quantity": 1, "unit": "potong" },
  { "food_name": "anggur",     "quantity": 1, "unit": "buah" }
]
```

> ⚠️ **PERHATIKAN:** `food_name` di `manual_items` **harus persis** sesuai hasil dari `/search-food`. Kalau tidak cocok, item tidak akan ketemu di database nutrisi.

**Contoh request (gambar saja):**
```bash
curl -X POST https://damassdev-nutrify-ai-api.hf.space/predict \
  -F "file=@nasi_goreng.jpg"
```

**Contoh request (gambar + manual + penyakit):**
```bash
curl -X POST https://damassdev-nutrify-ai-api.hf.space/predict \
  -F "image=@nasi_goreng.jpg" \
  -F 'manual_items=[{"food_name":"nasi putih","quantity":2,"unit":"porsi"}]' \
  -F "disease=diabetes"
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/coding-camp-project/AI.git
cd AI
```

### 2. Buat & aktifkan virtual environment
```bash
python -m venv venv
source venv/Scripts/activate     # Windows (Git Bash)
# source venv/bin/activate       # Linux / macOS
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> Kalau menambah library baru, update dengan: `pip freeze > requirements.txt`

### 4. Jalankan (lokal)
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```
> Sesuaikan `app.main:app` dengan entry point di `Dockerfile`.

### 5. Atau pakai Docker
```bash
docker build -t nutrify-ai-api .
docker run -p 7860:7860 nutrify-ai-api
```

### 6. Buka di Browser
```
http://localhost:7860/docs
```
> FastAPI otomatis menyediakan dokumentasi interaktif di `/docs` (Swagger UI).

# Selamat Datang di Nutrify AI 

## Model dan Backend yang disediakan oleh AI Engineer

> Kalau mau naro lib = pip freeze > requirements.txt
> Kalau mau install = pip install -r requirements.txt

1. python -m venv venv
2. source venv/Scripts/activate
3. python -m uvicorn app.main:app --reload


### Open to https://localhost:7680/docs

## Daftar Endpoint
  Method | Endpoint         | Fungsi
1. GET      `/`             Buat info dasar API
2. GET      `/health`       Buat helath check
3. GET      `/search-info`  Buat di cari makanan ada tidak di dataset (`?q=nama&limit=5`)
4. GET      `/units`        Daftar satuan porsi sama konveri gram
5. GET      `/disease`      Daftar penyakit untuk rekomendasi personalisasi di website
6. POST      `/predict`     Utama ini analsisa makann ada 3 input (gambar / manual / keduanya)


### Gambaran buat FS Nih POST /predict
- `image` - file foto makanan bisa .jpg, .webp, .png, dll masih satuan foto
- `manual_items` - WAJIB bentuk JSON array
```json
[{"food_name": "nasi putih", "quantity": 2, "unit": "porsi"}]
```

kalau mau nambah kek ada banyak item


```json
[
    {"food_name": "nasi putih", "quantity": 2, "unit": "porsi"}, 
    {"food_name": "Ayam Bakar", "quantity": 29, "unit": "potong"},
    {"food_name": "anggur", "quantity": 1, "unit": "buah"}
]
```
- `disease` - salah satu aja : `obesitas`, `diabetes`, `hipertensi`, `asam_urat`, `kolesterol


PERHATIKANN!!
Nama di `manual_items` harus **persis** sesuai hasil `/search-food`.

---
title: Nutrify AI API
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# Nutrify AI API
FastAPI service untuk prediksi makanan tradisional Indonesia.


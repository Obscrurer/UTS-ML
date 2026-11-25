# Kelompok 17 — UTS Machine Learning

Repository ini berisi submission untuk tiga task: regression, fraud classification, dan customer clustering.

## Struktur submission

```text
.
├── laporan.pdf
├── code/
│   ├── README.md
│   ├── 01_delivery_time_regression.ipynb
│   ├── 02_fraud_classification.ipynb
│   ├── 03_customer_clustering.ipynb
│   └── final_report.md
├── prediksi_regresi.csv
├── prediksi_klasifikasi.csv
├── cluster.csv
└── kontribusi.md
```

Nama file dan kolom prediksi mengikuti spesifikasi submission:
- `prediksi_regresi.csv`: `id_paket,p50,p90`
- `prediksi_klasifikasi.csv`: `id_transaksi,prob,tandai`
- `cluster.csv`: `id_pelanggan,cluster` (`-1` untuk noise/anomali)

## Menjalankan kode

Buka masing-masing notebook di folder `code/` melalui Google Colab. Letakkan `data_uts_ml.zip` di Google Drive pada `MyDrive/UTS_ML/`, lalu jalankan notebook dari atas ke bawah. Detail ada di `code/README.md`.

## Catatan penting

- Dataset mentah tidak disertakan dalam repository.
- Hasil prediksi yang disimpan berasal dari run notebook yang telah selesai.
- Fraud classifier menunjukkan penurunan PR-AUC yang cukup besar pada validasi Agustus; hal ini dijelaskan di laporan dan perlu disampaikan saat oral exam.
- Clustering menghasilkan dua segmen utama (aktif dan tidak aktif) serta label `-1` untuk noise.
- Lengkapi pemetaan nama/NIM, tautan fork dan pull request, serta persetujuan AI tools pada `kontribusi.md`.

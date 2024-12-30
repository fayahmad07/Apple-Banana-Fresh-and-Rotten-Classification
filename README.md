# Klasifikasi Apel dan Pisang: Segar dan Busuk

Repositori ini berisi implementasi Convolutional Neural Networks (CNN) menggunakan model yang telah dilatih sebelumnya untuk mengklasifikasikan apel dan pisang berdasarkan kondisi mereka: segar atau busuk.

## Deskripsi Proyek

Tujuan dari proyek ini adalah untuk mengembangkan model yang dapat membedakan antara apel dan pisang yang segar dengan yang busuk menggunakan teknik Deep Learning. Kami memanfaatkan arsitektur CNN yang telah dilatih sebelumnya, seperti VGG16, ResNet50, atau InceptionV3, untuk ekstraksi fitur yang efisien dan pelatihan model yang lebih cepat.

## Struktur Folder
.
├── data/                  # Dataset gambar
├── models/                # Model CNN yang disimpan
├── scripts/               # Skrip untuk pelatihan dan evaluasi model
├── results/               # Output evaluasi model, termasuk laporan klasifikasi
└── README.md              # Dokumentasi proyek


## Persiapan Lingkungan

Pastikan Anda memiliki Python dan library berikut terinstal:
- TensorFlow
- Keras
- NumPy
- Matplotlib

Anda dapat menginstal semua dependensi dengan menjalankan:
```bash
pip install tensorflow keras numpy matplotlib
Cara Menjalankan
Untuk melatih model, jalankan skrip pelatihan yang berada di direktori scripts:

bash
Salin kode
python scripts/train_model.py
Evaluasi Model
Setelah model dilatih, Anda dapat mengevaluasi performa model dengan menjalankan skrip evaluasi:

bash
Salin kode
python scripts/evaluate_model.py
Kontribusi
Kontribusi dari komunitas sangat dihargai. Jika Anda ingin berkontribusi:

Fork repositori ini.
Buat branch baru dengan fitur Anda (git checkout -b feature-baru).
Commit perubahan Anda (git commit -am 'Tambahkan fitur baru').
Push ke branch (git push origin feature-baru).
Buat pull request baru.

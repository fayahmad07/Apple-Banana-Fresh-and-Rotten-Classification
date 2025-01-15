# Klasifikasi Apel dan Pisang: Segar dan Busuk

Repositori ini berisi implementasi Convolutional Neural Networks (CNN) menggunakan model yang telah dilatih sebelumnya untuk mengklasifikasikan apel dan pisang berdasarkan kondisi mereka: segar atau busuk.

## Deskripsi Proyek

Tujuan dari proyek ini adalah untuk mengembangkan model yang dapat membedakan antara apel dan pisang yang segar dengan yang busuk menggunakan teknik Deep Learning. Kami memanfaatkan arsitektur CNN yang telah dilatih sebelumnya, seperti VGG16, ResNet50, atau InceptionV3, untuk ekstraksi fitur yang efisien dan pelatihan model yang lebih cepat.

## Latar Belakang

Mendeteksi kesegaran buah dan sayuran merupakan tantangan penting dalam industri pertanian. Identifikasi visual terhadap kualitas produk segar seringkali bergantung pada pengamatan manusia yang subjektif dan tidak konsisten. Kemajuan dalam bidang teknologi informasi, khususnya pembelajaran mesin dan pengolahan citra digital, telah membuka peluang untuk pengembangan sistem deteksi otomatis yang andal (Mukhiddinov et al., 2022).

## Tujuan dan Luaran Program

1.	Membandingkan tiga model pre-trained deep learning untuk 	mendeteksi tingkat kesegaran buah dan sayuran.
2.	Menganalisis kinerja dan hasil dari masing-masing model pre-trained 	yang diuji.
3.	Menguji model terpilih dengan data di luar dataset latih untuk 	mengevaluasi kinerja pada kondisi nyata.

## Algoritma Flowchart

![Final_PENMAN2FINAL_Flowchart drawio](https://github.com/user-attachments/assets/5216167d-f7f4-4edb-9f6d-c91dee552360)

## Convolutional Neural Network

CNN terdiri dari tiga lapisan utama:
1.  Convolutional Layer
Lapisan ini bertanggung jawab untuk mengekstraksi fitur-fitur penting dari data input, seperti tepi, tekstur, atau pola tertentu.
2.  Pooling Layer
Lapisan ini digunakan untuk mengurangi dimensi data, sehingga mengurangi kompleksitas komputasi tanpa kehilangan informasi penting.
3.  Fully-connected Layer
Lapisan ini menghubungkan seluruh neuron dari lapisan sebelumnya untuk membuat prediksi akhir berdasarkan fitur-fitur yang telah diekstraksi.

## Dataset

Dataset di ambil dari kaggle dengan total kelas 28 kelas yang digunakan dalam penelitian ini hanya 4 kelas yakni :
1.  Apple_Healty
2.  Apple_Rotten
3.  Banana_Healthy
4.  Banana_Rotten 

## Arsitektur VGG16

![image](https://github.com/user-attachments/assets/f42ac644-1c00-4bb9-9d84-d3473889502d)

## Arsitektur ResNet50

![image](https://github.com/user-attachments/assets/ec9c90a4-9f5b-4a4c-86d7-972df9dc3208)

## Arsitektur InceptionV3

![image](https://github.com/user-attachments/assets/9c4a7ecd-4271-4bb1-b45d-4bf721a8209d)

## Hasil

Penelitian ini menunjukkan bahwa pre-trained model yang mendapatkan hasil terbaik pada kasus ini yaitu inceptionV3 dengan akurasi sebesar 93% tidak hanya itu model juga sangat baik dalam mengenal data diluar dari dataset.

Performa model dalam pengujian data diluar dataset dapat dilihat pada file .ipynb

## Persiapan Library

Pastikan Anda memiliki Python dan library berikut terinstal:
- TensorFlow
- Keras
- NumPy
- Matplotlib



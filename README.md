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

https://viewer.diagrams.net/?tags=%7B%7D&lightbox=1&highlight=0000ff&edit=_blank&layers=1&nav=1&title=Final_PENMAN2FINAL_Flowchart.drawio#R%3Cmxfile%3E%3Cdiagram%20name%3D%22Final%20Flowchart%22%20id%3D%220%22%3EzZtLc6M4EIB%2FjY%2FZQgiwOebhTVI72U3FqZnklNIaxdYMlighJ%2Fb%2B%2BhVBMg%2BRKU%2BG0JyMGgmLr1tNdwsm%2BHyzu5QkW9%2BIhKYT30t2E3wx8X0UTQP9U0j2RhKjsJSsJEuMrBIs2H%2FUCD0j3bKE5o2OSohUsawpXArO6VI1ZERK8drs9izS5r9mZEUdwWJJUlf6jSVqXUpn%2FrSSX1G2Wtt%2FRlFcntkQ29ncSb4miXitifB8gs%2BlEKo82uzOaVrQs1zKcX%2B%2Bc%2FYwMUm5OmaAXw54IenW3NsXQRItuSCK5FSZWaq9vXUptjyhxWg0wWeva6boIiPL4uyr1raWrdUmNafd2ZgJvlCp6K4mMrO7pGJDldzrLuYs9gwpYyuBab5W3JGFua4xt%2F2IUfXqcOWKhj4wQLrhYAfOIkuZGjEdNCSep%2B%2FTnKM4ukv967unKxRePZKTYOpAoYlePKYppFqLleAknVfSswqbp1tVny9CZAbWd6rU3ngCslWiiZLumHqoHT8Wl%2FojNK2LnbnyW2NvG1zf70PVsWg%2B1s9Vw95adty7asvFVi7NPRsPp4hcUdPLgClo%2FFS1kqZEsZemq%2FkdPQWOGd9LwjjjKy1djMCKD37dWPGgizx06HwlKUu0CgQfCR8cA%2FKZOnzuaF6sQ9%2B73hTPSXA6LR%2BIrUqHoBM7dG4lPVHF%2BtIIxoYmwAMajl3U9ZV1eYkicCqzJpRwUCjIgaKX099UhR44l7a1DAvGjUSv%2BZJmhRv%2BisHZHFiAsHED0bcnuBaZ5GpklhMNSseNb8oneM5GC2g6KCA3xLmnuRotnNmgcCIHzrw4Lq3nSv%2BMD1A8KCA3ADwX%2FHmbl%2FHxDVGS7cARtVMI5A3KaOYySkmes2e2tInEHc10YgsOCs9AQbnx8l9io29ozMsNoSER2evWq05MI6o97b17Kv8l7Mf4SPmDknJj6YUiY1hjLSwn0ZBU3EB6zpPRMUF4UFNxM4heqpLvgqnXAg9bFLVioFHS8MVAWw2EIeGCwGAgQkAQ2AURgIFwKzKgIEIoENYh9Q3iUzcapi7AuG%2BAZuitYFzVPHrU9OiB1Zy9RDknM6qlhsM0jtOMD2iisUvYJhs9Iz6VkuxrHbICXf4LGohbm72t%2Fofdwe7%2B%2BqCcwYfVBPmU7VITAnMlkE%2FZLhK9xxuDGGwYf7LBQsYA1ok09NR7OPQhPbUdReT9XE9tvbb6%2F76eIEMU1BG%2B96%2Bnoz3LJ71fcRyKjgB%2BJCbbdhXgJjuD1FNHWI3AEgwcQ6LoeBMHgaUYwSelGMehCDtQRGAoECSKqAMF3AtakOlNJ4oZGArIFAJ1JNN2xwIABWQOYS1gJChA4%2FSOfMrvvQBwNArIUNjvSFn83lPLY1GcBsHT%2FW57e%2FrPjzPyQKbz7MY7se9R1bY%2FXGA8OS2%2BAtAtLjht7nU0WT0Lrsxbxzr4xWd5scFkBLO3il1Say238uWwp1KV%2FRpFv6oG%2BKtlv48USWubK2HH3oqV9VwK9KPWnk1pRE4p0LlQ0Eo8Q9S60IdrirpZfVVRdq8%2BTsHz%2FwE%3D%3C%2Fdiagram%3E%3C%2Fmxfile%3E

## Persiapan Library

Pastikan Anda memiliki Python dan library berikut terinstal:
- TensorFlow
- Keras
- NumPy
- Matplotlib



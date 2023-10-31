# Laporan Proyek Machine Learning
### Nama : Citra Rahmawati
### Nim : 211351037
### Kelas : TIF Pagi B

## Domain Proyek

Janin yang sehat sangat di dambakan oleh ibu hamil maupun keluarga. Kesehatan janin pun menjadi poin penting untuk keselamatan calon bayi maupun ibunya. Dalam menentukan apakah janin tersebut sehat atau tidak harus dilakukan dengan serius dan tidak boleh ada kesalahan guna keselamatan bayi dan ibu, selain itu untuk menentukan tindakan selanjutnya.

## Business Understanding

Pembuatan sitem prediksi kesehatan janin ini dapat mempermudah tim kesehatan dalam mengklasifikasi janin yang normal (sehat), suspect (diduga adanya gangguan) dan Pathological (terdapat masalah serius dalam kesehatan janin)

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Klasifikasi kesehatan janin yang masih manual

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mempermudah tim kesehatan dalam menentukan janin yang sehat atau yang memerlukan tindakan khusus 

    ### Solution statements
    - Pembuatan sistem yang membantu tim kesehatan baik dokter maupun perawat dalam menentukan tindakan dan prioritas tindakan kepada ibu hamil guna kesehatan janin dan kelahiran yang normal
    - Sistem yang dibuat menggunakan model yang menggunakan algoritma Logistic Regression

## Data Understanding
Pengurangan angka kematian anak tercermin dalam beberapa Tujuan Pembangunan Berkelanjutan PBB dan merupakan indikator utama kemajuan manusia.
PBB berharap bahwa pada tahun 2030, semua negara dapat mengakhiri kematian bayi baru lahir dan anak di bawah usia 5 tahun yang dapat dicegah, dan semua negara bertujuan untuk mengurangi kematian balita hingga setidaknya serendah 25 per 1.000 kelahiran.

Sejalan dengan pengertian kematian anak, tentu saja ada kematian ibu, yang menyumbang 295.000 kematian selama dan setelah kehamilan dan persalinan (per 2017). Sebagian besar kematian ini (94%) terjadi di negara dengan sumber daya yang terbatas, dan sebagian besar dapat dicegah.

Mengingat apa yang telah disebutkan di atas, Kardiotokogram (CTG) adalah pilihan yang sederhana dan dapat diakses dengan biaya yang terjangkau untuk menilai kesehatan janin, yang memungkinkan para profesional perawatan kesehatan untuk mengambil tindakan dalam rangka mencegah kematian anak dan ibu. Peralatan itu sendiri bekerja dengan mengirimkan denyut ultrasonik dan membaca responsnya, sehingga menjelaskan denyut jantung janin (FHR), gerakan janin, kontraksi rahim, dan banyak lagi.

Dataset ini berisi 2126 catatan fitur yang diekstrak dari pemeriksaan Kardiotokogram, yang kemudian diklasifikasikan oleh tiga dokter kandungan ahli ke dalam 3 kelas:

Normal
Tersangka (Suspect)
Patologis (Pathological)

Dataset: [Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- baseline value : Detak Jantung Janin Dasar (FHR)
- accelerations: Peningkatan tiba-tiba dan sementara dalam detak jantung janin
- fetal_movement : Jumlah gerakan janin per detik
- uterine_contractions : Jumlah kontraksi uterus per detik
- light_decelerations : Jumlah Dekelerasi ringan yang terjadi dalam satu detik pada janin selama periode waktu tertentu
- severe_decelerations : Penurunan tiba-tiba dan signifikan dalam detak jantung janin selama pemantauan elektronik kontinu (CTG) atau tes non-stress (non-stress test, NST)
- prolongued_decelerations : Penurunan tiba-tiba dan berkelanjutan dalam detak jantung janin yang berlangsung lebih lama dari yang dianggap normal
- abnormal_short_term_variability : Pola variabilitas yang tidak wajar atau tidak normal dalam detak jantung janin selama periode pendek
- mean_value_of_short_term_variability : Nilai rata-rata dari variabilitas pendek dalam detak jantung janin selama periode waktu tertentu
- percentage_of_time_with_abnormal_long_term_variability : Pada persentase waktu di mana variabilitas jangka panjang (long term variability) detak jantung janin menunjukkan pola yang tidak normal atau tidak sesuai dengan standar yang ditetapkan
- fetal_health : Kesehatan janin dengan kategori normal, suspect atau Pathological

deskripsi kolom fetal_health:
- Normal: Merujuk pada kondisi kesehatan janin yang sesuai dengan standar atau pola yang dianggap normal untuk usia kehamilan tertentu. Ini menunjukkan bahwa tidak ada tanda-tanda atau indikasi adanya masalah kesehatan pada janin.

- Suspect: Merujuk pada kondisi kesehatan janin yang memerlukan evaluasi lebih lanjut atau pemantauan lebih lanjut karena ada indikasi atau tanda-tanda yang tidak sepenuhnya normal. Ini mengindikasikan adanya kemungkinan adanya masalah atau kelainan pada janin, tetapi perlu dilakukan penilaian lebih lanjut untuk memastikannya.

- Pathological: Merujuk pada kondisi kesehatan janin yang menunjukkan adanya masalah atau kelainan yang signifikan. Ini menunjukkan adanya tanda-tanda yang jelas atau hasil tes yang menunjukkan adanya gangguan atau penyakit pada janin. Dalam kasus ini, diperlukan evaluasi, pengobatan, atau perawatan lebih lanjut untuk mengatasi kondisi patologis tersebut.


Seluruh kolom pada dataset tersebut memiliki tipe data ```float64```

## Data Preparation
Pertama kita cek dulu apakah ada data yang duplikat atau tidak:
```
df.duplicated().sum()
```
Mari kita hapus data yang duplikat dikarenakan terdapat 13 data yang duplikat
```
df = df.drop_duplicates()
```
setelah itu kita cek apakah ada data yang null:
```
df.isnull().sum()
```
```
sns.heatmap(df.isnull())
```
![image](https://github.com/citrarahma1/kesehatan-janin/assets/149367504/0185c858-220c-4b85-83fb-39db83c0bbf8)

Ternyata tidak ada data yang null

Sekarang mari kita cek penyebaran datasetnya
![image](https://github.com/citrarahma1/kesehatan-janin/assets/149367504/cd791fe3-4817-419e-8364-699f1d7c847b)

lalu menghapus kolom yang tidak akan dipakai:
```
df = df.drop(['mean_value_of_long_term_variability', 'histogram_width','histogram_min',
              'histogram_max', 'histogram_number_of_peaks','histogram_number_of_zeroes',
              'histogram_mode', 'histogram_mean','histogram_median', 'histogram_variance',
              'histogram_tendency'], axis=1)
```
selanjutnya kita lihat bagaimana visualisasi dari datasetnya:
```
df.hist(figsize = (20,20), color = "#5F9EA0")
```

![image](https://github.com/citrarahma1/kesehatan-janin/assets/149367504/910e288f-c8c8-42f9-979f-61a6e0dfbeaf)

## Modeling
Tahapan Modeling yang pertama adalah menentukan nilai X dan Y
```
X = df.drop (columns='fetal_health', axis=1)
Y = df['fetal_health']
```
selanjutnya membagi data training dan testing
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, stratify=Y, random_state=2)
```
Mendeklarasikan model yang dipakai, yaitu Logistic Regression
```
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Evaluation
Tahapan evaluasi yang dipakai dalam pembuatan model ini menggunakan metrik akurasi:
```
from sklearn.metrics import accuracy_score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```
```
print("Akurasi data training : ", training_data_accuracy)
```
Akurasi data training :  0.8236686390532545

Akurasi yang di dapatkan adalah 82% yang berarti model dapat dipakai.

## Deployment
[Prediksi Kesehatan Janin](https://kesehatan-janin-citra.streamlit.app/)


# Import library yang diperlukan
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_breast_cancer,load_wine

# page layout menjadi wide
st.set_page_config(page_title='Machine Learning App',layout='wide')

# membuat function untuk model building
def build_model(df):
  X = df.iloc[:,:-1]
  y = df.iloc[:,-1]
  
  # split data 
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=parameters_test_size,random_state=parameters_random_state)
  
  st.markdown('**Training Data Splitting**')
  st.write('Training X Set')
  st.info(X_train.shape)
  st.write('Testing X Set')
  st.info(X_test.shape)
  st.markdown('**Testing Data Splitting**')
  st.write('Training y Set')
  st.info(y_train.shape)
  st.write('Testing y Set')
  st.info(y_test.shape)
  
  st.markdown('**Variabels**')
  st.write('X Variabels')
  st.info(X.columns)
  st.write('Y Variabel')
  st.info(y.name)
  
  # model paramater using SVC(Support Vector Classifier) model
  lr = LogisticRegression(
    penalty = parameters_penalty,
    dual = parameters_dual,
    tol = parameters_tol,
    C = parameters_c,
    fit_intercept = parameters_fit_intercept,
    intercept_scaling = parameters_intercept_scaling,
    solver = parameters_solver,
    multi_class = parameters_multi_class,
    warm_start = parameters_warm_start,
    class_weight = parameters_class_weight,
    verbose = parameters_verbose,
    max_iter = parameters_max_iter,
    n_jobs = parameters_n_jobs,
    random_state = parameters_random_state,
    l1_ratio = parameters_l1_ratio
  )
  lr.fit(X_train,y_train)
  
  # Model performance
  st.subheader('Model Performance')
  
  st.markdown('**Training Set Performance**')
  y_pred_train = lr.predict(X_train)
  st.write('Coefficient of determination ($R^2$):')
  st.write("""Coefficient of determination ($R^2$) adalah regression score function
           Skor terbaik yang mungkin adalah 1,0 dan bisa negatif (karena modelnya bisa lebih buruk). Sebuah model konstan yang selalu memprediksi nilai yang diharapkan dari $R$, mengabaikan fitur input, akan mendapatkan skor 0,0.""")
  st.info(r2_score(y_train, y_pred_train))
  
  st.write('Accuracy Score Training Set')
  st.write("""
  Skor klasifikasi akurasi. Dalam klasifikasi multilabel, fungsi ini menghitung akurasi subset: kumpulan label yang diprediksi untuk sampel harus sama persis dengan kumpulan label yang sesuai di y_true.
  """)
  st.info(accuracy_score(y_train, y_pred_train))
  
  st.write('Classification Report Training Set')
  st.write("""
  Buat laporan teks yang menunjukkan metrik klasifikasi utama. Pada kolom pertama adalah skor precision, kolom kedua adalah skor recall, kolom ketiga adalah kolom f1_score, dan kolom keempat adalah skor support
  """)
  st.info(classification_report(y_train, y_pred_train))
  st.markdown('---')
  
  st.markdown('**Testing Performance**')
  y_pred_test = lr.predict(X_test)
  st.write('Coefficient of determination ($R^2$):')
  st.write("""Coefficient of determination ($R^2$) adalah regression score function
           Skor terbaik yang mungkin adalah 1,0 dan bisa negatif (karena modelnya bisa lebih buruk). Sebuah model konstan yang selalu memprediksi nilai yang diharapkan dari $R$, mengabaikan fitur input, akan mendapatkan skor 0,0.""")
  st.info(r2_score(y_test, y_pred_test))
  
  st.write('Accuracy Score Testing Set')
  st.write("""
  Skor klasifikasi akurasi. Dalam klasifikasi multilabel, fungsi ini menghitung akurasi subset: kumpulan label yang diprediksi untuk sampel harus sama persis dengan kumpulan label yang sesuai di y_true.
  """)
  st.info(accuracy_score(y_test, y_pred_test))
  
  st.write('Classification Report Testing Set')
  st.write("""
  Buat laporan teks yang menunjukkan metrik klasifikasi utama. Pada kolom pertama adalah skor precision, kolom kedua adalah skor recall, kolom ketiga adalah kolom f1_score, dan kolom keempat adalah skor support
  """)
  st.info(classification_report(y_test, y_pred_test))
  st.markdown('---')
  st.subheader('Model Parameters')
  st.write(lr.get_params())
  # lin = np.linspace(0,7,1000)
  # st.write('Hasil Prediksi dengan memasukkan data baru')
  # st.write([])
  
  # bar plot
  st.write('Bar Plot Coefficient of determination ($R^2$)')
  skor = [float(r2_score(y_train, y_pred_train)),float(r2_score(y_test, y_pred_test))]
  lab = ['r2_score train','r2_score test']
  dt = pd.DataFrame(skor,lab)
  st.bar_chart(dt,use_container_width=True)
  plt.ylabel('Score')
  plt.xlabel('Model')
  plt.title('Coefficient of determination ($R^2$)')
  plt.show()
  
  # bar plot mae mse
  st.write('Bar Plot Error MSE(Mean Squared Error) atau MAE(Mean Absolute Error)')
  skor2 = [float(accuracy_score(y_train, y_pred_train)),float(accuracy_score(y_test, y_pred_test))]
  lab2 = ['accuracy_score_train','accuracy_score_test']
  dt2 = pd.DataFrame(skor2,lab2)
  st.bar_chart(dt2,use_container_width=True)
  plt.ylabel('Score')
  plt.xlabel('Model')
  plt.title('Accuracy Score')
  plt.show()
  
  
  
  
# Halaman Utama
st.write("""
# Machine Learning Web App dengan Streamlit

Web ML App ini menggunakan model algoritma `LosgiticRegression`. 
`LogisticRegression` adalah teknik regresi yang fungsinya untuk memisahkan dataset menjadi dua bagian (kelompok). Seperti kasus koin yang dilempar, hasilnya hanya ada dua yaitu depan atau belakang. Begitu pula dengan regresi logistik, maka hasilnya hanyalah ada dua, yaitu YES atau No, atau bisa juga 1 atau 0. Contoh lainnya adalah mengklasifikasikan apakah e-mail spam atau bukan spam, klasifikasi apakah customer akan membeli atau tidak membeli.

Silahkan atur parameter untuk model `LogistikRegression`.
""")
st.write("""---""")

st.write("""
![gambar](https://raw.githubusercontent.com/AgungYogaSetiawan/portfolio/main/codedatawithyoga.png)
# Agung Yoga Setiawan 

Data Science Enthusiast Mahasiswa Semester 4 Teknik Informatika Salah Satu PTS Di Banjarmasin


## Social Media

 - Instagram: ***[ayogastwn_](https://instagram.com/ayogastwn_)***
 - Gmail: ***[agungyoga507@gmail.com](https://mail.google.com/mail/u/0/#inbox)***
 - Youtube: ***[Agung Yoga Setiawan](https://www.youtube.com/channel/UClTFB61ahqcBR1lHNFsQ00g)***

  
## Authors

- Github: [AgungYogaSetiawan](https://www.github.com/AgungYogaSetiawan)

""")
st.write("""---""")

st.write("""
# Keterangan Pengaturan Parameter
#### Silahkan dibaca terlebih dahulu untuk keterangan parameter model `LogisticRegression`, untuk dapat mengatur parameternya.

1. Parameter Penalty: Digunakan untuk menentukan norma yang digunakan dalam parameter penalty. Solver 'newton-cg', 'sag' dan 'lbfgs' hanya mendukung l2 penalti. 'elasticnet' hanya didukung oleh solver 'saga'. Jika 'None' (tidak didukung oleh solver liblinear), tidak ada regularisasi yang diterapkan.

2. Parameter dual: Formulasi ganda atau primal. Formulasi ganda hanya diterapkan untuk penalti l2 dengan pemecah liblinear. Lebih cocok dual=False ketika n_samples > n_features.

3. Parameter tol: Toleransi untuk menghentikan kriteria.

4. Parameter C: Kebalikan dari kekuatan regularisasi; harus menjadi float positif. Seperti pada model Support Vector Machine, nilai yang lebih kecil menentukan regularisasi yang lebih kuat.

5. Parameter fit_intercept: Menentukan apakah sebuah konstanta (alias bias atau intersep) harus ditambahkan ke fungsi keputusan.

6. Parameter intercept_scaling: Berguna hanya ketika solver 'liblinear' digunakan dan self.fit_intercept diatur ke True. Dalam hal ini, x menjadi [x, self.intercept_scaling], yaitu fitur “sintetis” dengan nilai konstan yang sama dengan intercept_scaling ditambahkan ke vektor instance. Intersep menjadi `intersep_scaling * synthetic_feature_weight`.
Catatan! bobot fitur sintetis tunduk pada regularisasi l1/l2 seperti semua fitur lainnya. Untuk mengurangi efek regularisasi pada bobot fitur sintetis (dan karenanya pada intersep) intersep_scaling harus ditingkatkan.

7. Parameter class_weight: Bobot yang terkait dengan kelas dalam bentuk `{class_label: weight}`. Jika tidak diberikan, semua kelas seharusnya memiliki bobot satu.
Mode “seimbang” menggunakan nilai y untuk menyesuaikan bobot secara otomatis berbanding terbalik dengan frekuensi kelas dalam data input sebagai `n_samples / (n_classes * np.bincount(y))`.
Perhatikan bahwa bobot ini akan dikalikan dengan sample_weight (melewati metode fit) jika sample_weight ditentukan.

8. Parameter random_state: Digunakan saat `solver` == 'sag', 'saga' atau 'liblinear' untuk mengacak data.

9. Parameter Solver: Algoritma yang akan digunakan dalam masalah optimasi.

  - Untuk kumpulan data kecil, 'liblinear' adalah pilihan yang baik, sedangkan 'sag' dan 'saga' lebih cepat untuk yang besar.

  - Untuk masalah multikelas, hanya 'newton-cg', 'sag', 'saga' dan 'lbfgs' yang menangani kerugian multinomial; 'liblinear' terbatas pada skema satu lawan satu istirahat.s.

  - 'newton-cg', 'lbfgs', 'sag' dan 'saga' menangani L2 atau tanpa penalti

  - 'liblinear' dan 'saga' juga menangani penalti L1

  - 'saga' juga mendukung penalti 'elasticnet'

  - 'liblinear' tidak mendukung pengaturan `penalti='none'`

 Perhatikan bahwa konvergensi cepat 'sag' dan 'saga' hanya dijamin pada fitur dengan skala yang kurang lebih sama. Anda dapat melakukan praproses data dengan scaler dari `sklearn.preprocessing`.
 
10. Parameter max_iter: Jumlah maksimum iterasi yang diambil untuk pemecah masalah untuk berkumpul.

11. Paramter multi_class: Jika opsi yang dipilih adalah 'ovr', maka masalah biner cocok untuk setiap label. Untuk 'multinomial' kerugian yang diminimalkan adalah kerugian multinomial yang cocok di seluruh distribusi probabilitas, bahkan ketika datanya biner. 'multinomial' tidak tersedia saat solver='liblinear'. 'auto' memilih 'ovr' jika datanya biner, atau jika solver='liblinear', dan sebaliknya memilih 'multinomial'.

12. Parameter verbose: Untuk solver 'liblinear' dan 'lbfgs' atur verbose ke bilangan positif apa pun untuk verbositas.

13. Parameter warm_start: Jika disetel ke True, gunakan kembali solusi dari panggilan sebelumnya agar sesuai sebagai inisialisasi, jika tidak, hapus saja solusi sebelumnya. Tidak berguna untuk solver 'liblinear'.

14. Parameter n_jobs: Jumlah core CPU yang digunakan saat memparalelkan kelas jika multi_class='ovr'”. Parameter ini diabaikan ketika solver diatur ke 'liblinear' terlepas dari apakah 'multi_class' ditentukan atau tidak. None berarti 1 kecuali dalam konteks ***joblib.parallel_backend***. -1 berarti menggunakan semua prosesor.

15. Parameter l1_ratio: Parameter pencampuran Elastic-Net, dengan `0 <= l1_ratio <= 1`. Hanya digunakan jika `penalty='elasticnet'`. Setting `l1_ratio=0` sama dengan menggunakan `penalty='l2'`, sedangkan setting `l1_ratio=1` sama dengan menggunakan `penalty='l1'`. Untuk `0 < l1_ratio <1`, penaltinya adalah kombinasi dari L1 dan L2.
""")

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('Upload CSV Data Hanya untuk Clasification'):
  uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
  st.sidebar.markdown("""
[Download Wine Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)
""")

# Sidebar untuk mengatur parameter model
with st.sidebar.header('Set Parameters Logistic Regression'):
  parameters_test_size = st.sidebar.slider('Data split ratio (for Training Set)', 10, 90, 80, 5)
  parameters_penalty = st.sidebar.selectbox('Penalty (default="l2")',options=['l2','l1','elasticnet','none'])
  parameters_dual = st.sidebar.selectbox('Pilih Parameter Dual',(False,True))
  parameters_tol = st.sidebar.slider('Pilih Tolerance',0.0001)
  parameters_c = st.sidebar.slider('Pilih C',0.5,2.0,1.0)
  parameters_fit_intercept = st.sidebar.selectbox('Pilih Fit Intercept',(True,False))
  parameters_intercept_scaling = st.sidebar.slider('intercept_scaling', 0.1,0.9,1.0)
  parameters_solver = st.sidebar.select_slider('Pilih Solver',options=['lbfgs','newton-cg','liblinear','sag','saga'])
  parameters_multi_class = st.sidebar.select_slider('Pilih Multi Class',['auto','ovr','multinomial'])
  parameters_warm_start = st.sidebar.selectbox('Pilih Wrm Start',(True,False))
  parameters_class_weight = st.sidebar.selectbox('Pilih Class Weight',[None,'balanced'])
  parameters_verbose = st.sidebar.slider('Pilih Verbose',0,100,0)
  parameters_max_iter = st.sidebar.slider('Pilih Max ter',0,100,100)
  parameters_n_jobs = st.sidebar.slider('Pilih n_jobs',1,10)
  parameters_random_state = st.sidebar.slider('Pilih Random State',0,42)
  parameters_l1_ratio = st.sidebar.selectbox('Pilih l1_ratio',[None])

# Main Menu
st.subheader('Dataset')

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.markdown('**Isi Dataset**')
  st.write(df)
  build_model(df)
else:
  if st.button('Demo Iris Dataset'):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    df = pd.concat( [X,y], axis=1 )
    st.write('Memakai Dataset Iris')
    st.write(df.head())
    
    # # membuat user input untuk prediksi target names
    # sl = st.number_input('Masukan angka untuk sepal length',min_value=0,max_value=30)
    # sw = st.number_input('Masukan angka untuk sepal width',min_value=0,max_value=30)
    # pl = st.number_input('Masukan angka untuk petal length',min_value=0,max_value=30)
    # pw = st.number_input('Masukan angka untuk petal width',min_value=0,max_value=30)
    # wrap = [sl,sw,pl,pw]
    # st.info(X.predict(wrap))
    
    build_model(df)
  elif st.button('Demo Breast Cancer Dataset'):
    bc = load_breast_cancer()
    X = pd.DataFrame(bc.data, columns=bc.feature_names)
    y = pd.Series(bc.target)
    df = pd.concat( [X,y], axis=1 )
    st.write('Memakai Dataset Breast Cancer')
    st.write(df.head())
    
    build_model(df)

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load Model and Scaler
# Ensure these files are in the same directory as app.py or provide the full path
try:
    with open('best_gradient_boosting_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('standard_scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)
    st.success("Model and Scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Make sure 'best_gradient_boosting_model.pkl' and 'standard_scaler.pkl' are in the correct directory.")
    st.stop() # Stop the app if files are not found

# --- Hardcoded Mappings and Column Order (from training) ---
# This is crucial for consistent preprocessing

# Gender mapping (from ZfaZ4X9NrssK)
mapping_gender = {'Pria':'Laki-laki', 'L':'Laki-laki', 'Perempuan' : 'Wanita', 'P':'Wanita'}

# Example of `df_bersih`'s unique values for LabelEncoder fitting
# In a real app, these would be saved alongside the model/scaler or derived from a small sample of training data
# For this example, we'll use assumed unique values based on the notebook's execution
# from v9ubzlF7dChe and I0mkz4u6gHmg

# For 'Pendidikan'
original_pendidikan_unique = ['SMA', 'SMK', 'D3', 'S1'] # Order might matter for LabelEncoder consistency if not explicitly fitted
le_pendidikan = LabelEncoder()
le_pendidikan.fit(original_pendidikan_unique)

# For 'Jurusan'
original_jurusan_unique = ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif'] # Order might matter
le_jurusan = LabelEncoder()
le_jurusan.fit(original_jurusan_unique)

# Original feature columns from X_train (from 0ZVos9Vl2ESv, H85GRhKhyOUF)
# This is the order and exact names the model expects after all preprocessing
original_feature_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
                         'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                         'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']

# Streamlit App Title
st.title('Salary Prediction App for Vocational Training Participants')
st.write('Enter participant details to predict their initial salary.')

# --- User Input Fields ---

# Numerical Inputs
usia = st.slider('Usia (Tahun)', 18, 60, 27) # Adjusted max age based on outlier removal (batas_atas_usia approx 59)
durasi_jam = st.slider('Durasi Pelatihan (Jam)', 20, 100, 50)
nilai_ujian = st.slider('Nilai Ujian (0-100)', 0.0, 100.0, 85.0)

# Categorical Inputs
pendidikan = st.selectbox('Pendidikan', ['SMA', 'SMK', 'D3', 'S1'])
jurusan = st.selectbox('Jurusan', ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif'])
jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Wanita'])
status_bekerja = st.selectbox('Status Bekerja', ['Sudah Bekerja', 'Belum Bekerja'])

# --- Prediction Button ---
if st.button('Predict Salary'):
    # 1. Create a DataFrame from user inputs
    input_df = pd.DataFrame({
        'Usia': [usia],
        'Durasi_Jam': [durasi_jam],
        'Nilai_Ujian': [nilai_ujian],
        'Pendidikan': [pendidikan],
        'Jurusan': [jurusan],
        'Jenis_Kelamin': [jenis_kelamin],
        'Status_Bekerja': [status_bekerja]
    })

    # 2. Preprocessing - Follow the exact steps from the notebook

    # a. Apply gender mapping
    input_df['Jenis_Kelamin'] = input_df['Jenis_Kelamin'].replace(mapping_gender)

    # b. Label Encoding for 'Pendidikan' and 'Jurusan'
    df_label_new = pd.DataFrame()
    df_label_new['Pendidikan'] = le_pendidikan.transform(input_df['Pendidikan'])
    df_label_new['Jurusan'] = le_jurusan.transform(input_df['Jurusan'])

    # c. One-Hot Encoding for 'Jenis_Kelamin' and 'Status_Bekerja'
    # Ensure consistency in columns with training data
    df_onehot_new = pd.DataFrame(0, index=input_df.index, columns=[
        'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
    ])
    
    # Fill in the values for the existing columns
    if (input_df['Jenis_Kelamin'] == 'Laki-laki').any():
        df_onehot_new['Jenis_Kelamin_Laki-laki'] = 1
    else:
        df_onehot_new['Jenis_Kelamin_Wanita'] = 1

    if (input_df['Status_Bekerja'] == 'Belum Bekerja').any():
        df_onehot_new['Status_Bekerja_Belum Bekerja'] = 1
    else:
        df_onehot_new['Status_Bekerja_Sudah Bekerja'] = 1

    # d. Extract numerical columns
    df_numerik_new = input_df[['Usia', 'Durasi_Jam', 'Nilai_Ujian']]

    # e. Combine all processed features
    unscaled_processed_data = pd.concat([df_numerik_new, df_label_new, df_onehot_new], axis=1)

    # f. Align columns and scale
    processed_data_aligned = unscaled_processed_data[original_feature_cols]
    scaled_processed_data = loaded_scaler.transform(processed_data_aligned)
    scaled_processed_df = pd.DataFrame(scaled_processed_data, columns=original_feature_cols)

    # 3. Make Prediction
    predicted_gaji = loaded_model.predict(scaled_processed_df)

    st.success(f"Predicted Initial Salary: {predicted_gaji[0]:.2f} Juta Rupiah")

# Optional: Display raw input for debugging
st.sidebar.header('Raw Input Data')
st.sidebar.write(pd.DataFrame({
    'Usia': [usia],
    'Durasi_Jam': [durasi_jam],
    'Nilai_Ujian': [nilai_ujian],
    'Pendidikan': [pendidikan],
    'Jurusan': [jurusan],
    'Jenis_Kelamin': [jenis_kelamin],
    'Status_Bekerja': [status_bekerja]
}))

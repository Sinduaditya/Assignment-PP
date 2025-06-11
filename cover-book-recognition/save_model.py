import easyocr
import torch
import numpy as np
import h5py
import os

def save_model_to_h5():
    # Inisialisasi reader dengan model yang diinginkan
    reader = easyocr.Reader(['en'], gpu=True)
    
    # Buat direktori untuk menyimpan model jika belum ada
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    
    # Simpan model recognition
    recognition_model = reader.recognizer
    recognition_model.eval()  # Set ke mode evaluasi
    
    # Simpan model detection
    detection_model = reader.detector
    detection_model.eval()
    
    # Simpan model recognition
    with h5py.File('saved_models/recognition_model.h5', 'w') as f:
        # Simpan state dict
        for name, param in recognition_model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())
    
    # Simpan model detection
    with h5py.File('saved_models/detection_model.h5', 'w') as f:
        # Simpan state dict
        for name, param in detection_model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())
    
    print("Model berhasil disimpan di folder 'saved_models'")
    print("1. recognition_model.h5 - Model untuk pengenalan teks")
    print("2. detection_model.h5 - Model untuk deteksi teks")

if __name__ == "__main__":
    save_model_to_h5() 
import easyocr
import cv2
import numpy as np
import torch
import os
import json
from evaluate_model import ModelEvaluator
import time

class ModelTuner:
    def __init__(self, model_path='saved_models'):
        self.model_path = model_path
        self.evaluator = ModelEvaluator(model_path)
        
    def tune_parameters(self, dataset_image_path, dataset_gt_path, param_combinations):
        """
        Melakukan hyperparameter tuning untuk model OCR
        
        Parameters:
        - dataset_image_path: path ke folder gambar training
        - dataset_gt_path: path ke folder ground truth
        - param_combinations: list of parameter dictionaries to try
        """
        best_score = 0
        best_params = None
        results = []
        
        total_combinations = len(param_combinations)
        
        print(f"\nMemulai hyperparameter tuning dengan {total_combinations} kombinasi parameter...")
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nMencoba kombinasi {i}/{total_combinations}")
            print("Parameter:", params)
            
            # Inisialisasi reader dengan parameter baru
            reader = easyocr.Reader(
                ['en'],
                gpu=torch.cuda.is_available(),
                **params
            )
            
            # Evaluasi model dengan parameter ini
            start_time = time.time()
            metrics = self.evaluator.evaluate_dataset(dataset_image_path, dataset_gt_path)
            evaluation_time = time.time() - start_time
            
            # Hitung score (gunakan F1-score sebagai metrik utama)
            score = metrics['avg_f1_score']
            
            # Simpan hasil
            result = {
                'parameters': params,
                'metrics': metrics,
                'evaluation_time': evaluation_time
            }
            results.append(result)
            
            print(f"F1-Score: {score:.3f}")
            print(f"Waktu evaluasi: {evaluation_time:.2f} detik")
            
            # Update best parameters jika score lebih baik
            if score > best_score:
                best_score = score
                best_params = params
                print("*** Parameter baru terbaik ditemukan! ***")
        
        # Simpan hasil tuning ke file
        tuning_results = {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results
        }
        
        with open('tuning_results.json', 'w') as f:
            json.dump(tuning_results, f, indent=4)
        
        return best_params, best_score, results

def main():
    # Definisikan 2 kombinasi parameter untuk komputasi rendah
    param_combinations = [
        # Kombinasi 1: Default settings dengan optimasi untuk komputasi rendah
        {
            'model_storage_directory': 'saved_models',
            'download_enabled': True,
            'user_network_directory': 'saved_models',
            'recog_network': 'standard'
        },
        # Kombinasi 2: Network yang lebih ringan dengan optimasi
        {
            'model_storage_directory': 'saved_models',
            'download_enabled': True,
            'user_network_directory': 'saved_models',
            'recog_network': 'latin_g1'  # Network yang lebih ringan
        }
    ]
    
    # Path ke dataset
    dataset_image_path = "archive/ch4_training_images"
    dataset_gt_path = "archive/ch4_training_localization_transcription_gt"
    
    if not os.path.exists(dataset_image_path) or not os.path.exists(dataset_gt_path):
        print("Error: Dataset tidak ditemukan!")
        print(f"Pastikan folder dataset ada di {dataset_image_path} dan {dataset_gt_path}")
        return
    
    # Inisialisasi tuner
    tuner = ModelTuner()
    
    # Mulai tuning
    print("\nMemulai proses hyperparameter tuning...")
    best_params, best_score, results = tuner.tune_parameters(
        dataset_image_path,
        dataset_gt_path,
        param_combinations
    )
    
    # Tampilkan hasil
    print("\nHasil Tuning:")
    print(f"Parameter Terbaik: {best_params}")
    print(f"Score Terbaik (F1-Score): {best_score:.3f}")
    print("\nHasil lengkap telah disimpan ke 'tuning_results.json'")

if __name__ == "__main__":
    main() 
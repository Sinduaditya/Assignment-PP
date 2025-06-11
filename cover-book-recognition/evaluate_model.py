import easyocr
import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os
import h5py
import torch

class ModelEvaluator:
    def __init__(self, model_path='saved_models'):
        # Inisialisasi reader
        # Menggunakan gpu=True jika tersedia, jika tidak, akan default ke CPU
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Load model jika ada
        if os.path.exists(model_path):
            try:
                # Load model recognition
                with h5py.File(f'{model_path}/recognition_model.h5', 'r') as f:
                    state_dict = {}
                    for name in f.keys():
                        data = f[name][()]
                        if isinstance(data, np.ndarray):
                            state_dict[name] = torch.from_numpy(data)
                        else:
                            state_dict[name] = torch.tensor(data)
                    self.reader.recognizer.load_state_dict(state_dict)
                
                # Load model detection
                with h5py.File(f'{model_path}/detection_model.h5', 'r') as f:
                    state_dict = {}
                    for name in f.keys():
                        data = f[name][()]
                        if isinstance(data, np.ndarray):
                            state_dict[name] = torch.from_numpy(data)
                        else:
                            state_dict[name] = torch.tensor(data)
                    self.reader.detector.load_state_dict(state_dict)
                
                print("Model berhasil dimuat dari", model_path)
            except Exception as e:
                print(f"Error saat memuat model: {str(e)}")
                print("Menggunakan model default EasyOCR")
    
    def parse_icdar_gt(self, gt_path):
        """
        Parse ground truth dari format ICDAR (.txt)
        """
        ground_truth = []
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {gt_path}")
            return ground_truth
            
        try:
            with open(gt_path, 'r', encoding='utf-8-sig') as f:  # Menggunakan utf-8-sig untuk handle BOM
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split berdasarkan koma, 8 pertama adalah bbox, sisanya teks
                    parts = line.split(',')
                    if len(parts) < 9:
                        print(f"Warning: Invalid ground truth line in {gt_path}: {line}")
                        continue
                    
                    try:
                        # Ambil koordinat bbox dan bersihkan dari karakter non-numerik
                        bbox_coords = []
                        for p in parts[:8]:
                            # Bersihkan karakter non-numerik dan BOM
                            clean_p = ''.join(c for c in p if c.isdigit() or c == '-')
                            if clean_p:
                                bbox_coords.append(int(clean_p))
                            else:
                                raise ValueError(f"Invalid coordinate value: {p}")
                        
                        if len(bbox_coords) != 8:
                            raise ValueError(f"Invalid number of coordinates: {len(bbox_coords)}")
                        
                        bbox = []
                        for i in range(0, 8, 2):
                            bbox.append([bbox_coords[i], bbox_coords[i+1]])
                            
                        # Ambil teks (gabungkan sisa bagian setelah 8 koma pertama)
                        text = ','.join(parts[8:])
                        
                        # Hapus BOM dan karakter khusus lainnya
                        text = text.encode('ascii', 'ignore').decode('ascii')
                        
                        # ICDAR menggunakan '###' untuk ignore regions
                        if text == '###':
                            continue # Abaikan teks ini jika itu adalah ignore region

                        ground_truth.append({'text': text, 'bbox': bbox})
                    except ValueError as e:
                        print(f"Warning: Could not parse bbox coordinates in {gt_path}: {line} - {e}")
                        continue
        except Exception as e:
            print(f"Error reading ground truth file {gt_path}: {e}")
            return ground_truth

        return ground_truth
        
    def evaluate_single_image(self, image_path, ground_truth_path):
        """
        Evaluasi model pada satu gambar
        image_path: path ke gambar
        ground_truth_path: path ke file ground truth ICDAR (.txt)
        """
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        # Parse ground truth
        ground_truth = self.parse_icdar_gt(ground_truth_path)

        # Lakukan OCR
        results = self.reader.readtext(image)
        
        # Hitung metrik
        metrics = {
            'detected_texts': [],
            'ground_truth_texts': [],
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'confidence_scores': []
        }
        
        # Kumpulkan teks yang terdeteksi
        detected_texts = [text for _, text, _ in results]
        ground_truth_texts = [gt['text'] for gt in ground_truth]
        
        # Hitung metrik
        # Menggunakan karakter level accuracy
        if detected_texts and ground_truth_texts:
            # Gabungkan semua teks ground truth
            all_gt_text = ' '.join(ground_truth_texts).lower()
            # Gabungkan semua teks yang terdeteksi
            all_detected_text = ' '.join(detected_texts).lower()
            
            # Buat set karakter unik
            all_chars = sorted(list(set(all_gt_text + all_detected_text)))
            
            # Buat binary vectors untuk setiap karakter
            y_true = [1 if char in all_gt_text else 0 for char in all_chars]
            y_pred = [1 if char in all_detected_text else 0 for char in all_chars]
            
            # Hitung metrik
            if all_chars:
                 metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                 metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
                 metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Simpan hasil
        metrics['detected_texts'] = detected_texts
        metrics['ground_truth_texts'] = ground_truth_texts
        metrics['confidence_scores'] = [prob for _, _, prob in results]
        
        return metrics
    
    def evaluate_dataset(self, dataset_image_path, dataset_gt_path):
        """
        Evaluasi model pada dataset ICDAR
        dataset_image_path: path ke folder gambar training/testing ICDAR
        dataset_gt_path: path ke folder ground truth training/testing ICDAR
        """
        all_metrics = []
        
        # Baca semua gambar di folder
        for filename in os.listdir(dataset_image_path):
            if filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(dataset_image_path, filename)
                # Cari file ground truth yang sesuai
                # Nama file ground truth ICDAR: gt_[nama_gambar_tanpa_ekstensi].txt
                gt_filename = f"gt_{os.path.splitext(filename)[0]}.txt"
                gt_path = os.path.join(dataset_gt_path, gt_filename)
                
                # Evaluasi gambar jika file ground truth ada
                metrics = self.evaluate_single_image(image_path, gt_path)
                all_metrics.append(metrics)
        
        # Hitung rata-rata metrik
        avg_metrics = {
            'avg_precision': np.mean([m['precision'] for m in all_metrics]) if all_metrics else 0,
            'avg_recall': np.mean([m['recall'] for m in all_metrics]) if all_metrics else 0,
            'avg_f1_score': np.mean([m['f1_score'] for m in all_metrics]) if all_metrics else 0,
            'avg_confidence': np.mean([conf for m in all_metrics for conf in m['confidence_scores']]) if any(m['confidence_scores'] for m in all_metrics) else 0
        }
        
        return avg_metrics

def main():
    # Contoh penggunaan
    evaluator = ModelEvaluator()
    
    # Path ke folder gambar dan ground truth training ICDAR
    icdar_train_images_path = "archive/ch4_training_images"
    icdar_train_gt_path = "archive/ch4_training_localization_transcription_gt"
    
    if os.path.exists(icdar_train_images_path) and os.path.exists(icdar_train_gt_path):
        print(f"\nEvaluasi menggunakan dataset training ICDAR dari {icdar_train_images_path} dan {icdar_train_gt_path}")
        avg_metrics = evaluator.evaluate_dataset(icdar_train_images_path, icdar_train_gt_path)
        print("\nHasil Evaluasi Dataset Training ICDAR:")
        print(f"Rata-rata Precision: {avg_metrics['avg_precision']:.3f}")
        print(f"Rata-rata Recall: {avg_metrics['avg_recall']:.3f}")
        print(f"Rata-rata F1-Score: {avg_metrics['avg_f1_score']:.3f}")
        print(f"Rata-rata Confidence: {avg_metrics['avg_confidence']:.3f}")
    else:
        print(f"Folder dataset ICDAR training tidak ditemukan di {icdar_train_images_path} dan {icdar_train_gt_path}.")
        print("Pastikan Anda sudah mengunduh dan mengekstrak dataset ICDAR.")

if __name__ == "__main__":
    main() 
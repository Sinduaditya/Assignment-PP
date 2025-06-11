import easyocr
import torch
import numpy as np
import h5py
import cv2

class CustomEasyOCR:
    def __init__(self, model_path='saved_models'):
        # Initialize reader with default model
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        try:
            # Load recognition model
            with h5py.File(f'{model_path}/recognition_model.h5', 'r') as f:
                state_dict = {}
                for name in f.keys():
                    data = f[name][()]
                    if isinstance(data, np.ndarray):
                        state_dict[name] = torch.from_numpy(data)
                    else:
                        state_dict[name] = torch.tensor(data)
                self.reader.recognizer.load_state_dict(state_dict)
            
            # Load detection model
            with h5py.File(f'{model_path}/detection_model.h5', 'r') as f:
                state_dict = {}
                for name in f.keys():
                    data = f[name][()]
                    if isinstance(data, np.ndarray):
                        state_dict[name] = torch.from_numpy(data)
                    else:
                        state_dict[name] = torch.tensor(data)
                self.reader.detector.load_state_dict(state_dict)
            
            print("Model berhasil dimuat!")
        except Exception as e:
            print(f"Error saat memuat model: {str(e)}")
            print("Menggunakan model default EasyOCR")
    
    def readtext(self, image):
        """
        Membaca teks dari gambar menggunakan model yang telah dimuat
        """
        return self.reader.readtext(image)

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    
    # Buat instance CustomEasyOCR
    ocr = CustomEasyOCR()
    
    try:
        while True:
            # Baca frame dari kamera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            
            # Resize frame untuk performa lebih baik
            frame = cv2.resize(frame, (640, 480))
            
            # Lakukan OCR pada frame
            results = ocr.readtext(frame)
            
            # Gambar hasil deteksi
            for (bbox, text, prob) in results:
                # Konversi koordinat bbox ke numpy array dengan tipe int32
                bbox = np.array(bbox, dtype=np.int32)
                # Gambar kotak
                cv2.polylines(frame, [bbox], True, (0, 255, 0), 2)
                # Tampilkan teks dan confidence
                text_with_conf = f"{text} ({prob:.2f})"
                cv2.putText(frame, text_with_conf, (bbox[0][0], bbox[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Tampilkan frame
            cv2.imshow('Custom EasyOCR', frame)
            
            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Bersihkan
        cap.release()
        cv2.destroyAllWindows() 
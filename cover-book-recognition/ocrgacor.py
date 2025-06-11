import cv2
import easyocr
import numpy as np
import torch
import h5py
import os

class OCRCamera:
    def __init__(self, model_path='saved_models'):
        # Inisialisasi reader dengan model default
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Load model yang sudah disimpan
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
        
        # Inisialisasi kamera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Tidak dapat membuka kamera")
        
        # Set resolusi kamera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def process_frame(self, frame):
        """Proses frame untuk deteksi dan pengenalan teks"""
        # Lakukan OCR pada frame
        results = self.reader.readtext(frame)
        
        # Gambar hasil deteksi
        output_frame = frame.copy()
        
        # Untuk setiap teks yang terdeteksi
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Hanya tampilkan teks dengan confidence > 50%
                # Konversi koordinat bbox ke integer
                bbox = np.array(bbox).astype(np.int32)
                
                # Gambar bounding box
                cv2.polylines(output_frame, [bbox], True, (0, 255, 0), 2)
                
                # Tambahkan teks dan confidence score
                text_position = (bbox[0][0], bbox[0][1] - 10)
                cv2.putText(output_frame, f"{text} ({prob:.2f})", text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return output_frame, results
    
    def run(self):
        """Jalankan kamera dan proses OCR"""
        print("Tekan 'q' untuk keluar")
        print("Tekan 'c' untuk capture dan simpan gambar")
        
        while True:
            # Baca frame dari kamera
            ret, frame = self.cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera")
                break
            
            # Proses frame
            output_frame, results = self.process_frame(frame)
            
            # Tampilkan hasil
            cv2.imshow('OCR Camera', output_frame)
            
            # Tunggu input keyboard
            key = cv2.waitKey(1) & 0xFF
            
            # Keluar jika 'q' ditekan
            if key == ord('q'):
                break
            
            # Capture dan simpan jika 'c' ditekan
            elif key == ord('c'):
                # Simpan gambar asli
                cv2.imwrite('captured_image.jpg', frame)
                # Simpan gambar dengan hasil OCR
                cv2.imwrite('captured_image_with_ocr.jpg', output_frame)
                print("Gambar berhasil disimpan!")
                
                # Tampilkan teks yang terdeteksi
                print("\nTeks yang terdeteksi:")
                for (_, text, prob) in results:
                    if prob > 0.5:
                        print(f"- {text} (confidence: {prob:.2f})")
        
        # Bersihkan
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        ocr_camera = OCRCamera()
        ocr_camera.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import torch
import h5py
import easyocr
import os
import time

class OCRApp:
    def __init__(self, model_path='saved_models'):
        # Initialize reader with default model
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Load saved model
        if os.path.exists(model_path):
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
                
                st.success("Model berhasil dimuat!")
            except Exception as e:
                st.error(f"Error saat memuat model: {str(e)}")
                st.info("Menggunakan model default EasyOCR")

    def process_image(self, image):
        """Process image for text detection and recognition"""
        # Convert PIL Image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Perform OCR
        results = self.reader.readtext(image)
        
        # Draw results on image
        output_image = image.copy()
        detected_texts = []
        
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Only show text with confidence > 50%
                # Convert bbox coordinates to integer
                bbox = np.array(bbox).astype(np.int32)
                
                # Draw bounding box
                cv2.polylines(output_image, [bbox], True, (0, 255, 0), 2)
                
                # Add text and confidence score
                text_position = (bbox[0][0], bbox[0][1] - 10)
                cv2.putText(output_image, f"{text} ({prob:.2f})", text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                detected_texts.append((text, prob))
        
        # Convert back to RGB for display
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        return output_image, detected_texts

def main():
    st.title("OCR Text Recognition App")
    st.write("Upload an image or use your webcam to detect and recognize text")

    # Initialize OCR
    ocr = OCRApp()

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Webcam"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Process Image"):
                with st.spinner("Processing..."):
                    # Process the image
                    output_image, detected_texts = ocr.process_image(image)
                    
                    # Display results
                    st.image(output_image, caption="Processed Image", use_column_width=True)
                    
                    # Display detected texts
                    if detected_texts:
                        st.subheader("Detected Texts:")
                        for text, prob in detected_texts:
                            st.write(f"- {text} (confidence: {prob:.2f})")
                    else:
                        st.info("No text detected in the image")

    with tab2:
        st.write("Click 'Start Webcam' to begin real-time text detection")
        
        # Initialize webcam
        if 'webcam_on' not in st.session_state:
            st.session_state.webcam_on = False
        
        # Create columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Webcam"):
                st.session_state.webcam_on = True
        
        with col2:
            if st.button("Stop Webcam"):
                st.session_state.webcam_on = False
        
        # Webcam feed
        if st.session_state.webcam_on:
            # Create a placeholder for the webcam feed
            webcam_placeholder = st.empty()
            results_placeholder = st.empty()
            
            try:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                while st.session_state.webcam_on:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break
                    
                    # Convert frame to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    output_frame, detected_texts = ocr.process_image(Image.fromarray(frame_rgb))
                    
                    # Display the frame
                    webcam_placeholder.image(output_frame, channels="RGB", use_column_width=True)
                    
                    # Display detected texts
                    if detected_texts:
                        text_output = "**Detected Texts:**\n"
                        for text, prob in detected_texts:
                            text_output += f"- {text} (confidence: {prob:.2f})\n"
                        results_placeholder.markdown(text_output)
                    else:
                        results_placeholder.info("No text detected")
                    
                    # Add a small delay to prevent overwhelming the system
                    time.sleep(0.1)
                
                cap.release()
                
            except Exception as e:
                st.error(f"Error accessing webcam: {str(e)}")
                st.session_state.webcam_on = False

if __name__ == "__main__":
    main() 
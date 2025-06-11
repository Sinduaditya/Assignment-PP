from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
from load_model import CustomEasyOCR

app = Flask(__name__)
CORS(app)

# Load the OCR model
model = CustomEasyOCR()

# Load book categories
with open('book_categories.json', 'r', encoding='utf-8') as f:
    book_categories = json.load(f)

def categorize_text(text):
    text = text.lower()
    best_match = None
    highest_score = 0
    
    for category in book_categories:
        score = 0
        matched_keywords = []
        for keyword in category['keywords']:
            if keyword.lower() in text:
                score += 1
                matched_keywords.append(keyword)
        
        # Calculate match percentage based on total keywords
        match_percentage = (score / len(category['keywords'])) * 100
        
        if score > highest_score:
            highest_score = score
            best_match = {
                'category_name': category['category_name'],
                'rack_id': category['rack_id'],
                'match_score': score,
                'match_percentage': match_percentage,
                'matched_keywords': matched_keywords,
                'total_keywords': len(category['keywords'])
            }
    
    return [best_match] if best_match else []

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})

        # Decode the base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image with OCR
        results = model.readtext(image)
        
        # Process results and create output image
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
                
                detected_texts.append({"text": text, "confidence": float(prob)})
        
        # Convert the processed image to base64
        _, buffer = cv2.imencode('.jpg', output_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Categorize the detected text
        combined_text = ' '.join([item['text'] for item in detected_texts])
        categories = categorize_text(combined_text)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{processed_image_base64}',
            'texts': detected_texts,
            'categories': categories
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
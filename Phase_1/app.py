from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to connect

# Load the trained model
print("Loading model...")
model = load_model("mobilenetv2_potato_binary.keras")
print("✅ Model loaded successfully!")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize to 224x224 (MobileNetV2 input size)
    img = cv2.resize(img, (224, 224))
    # Normalize to [0, 1]
    img = img / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return jsonify({
        "message": "🥔 Potato Classifier API",
        "status": "running",
        "endpoints": {
            "/classify": "POST - Upload image for classification"
        }
    })

@app.route('/classify', methods=['POST'])
def classify():
    print("📥 Received classification request")
    
    # Check if image was uploaded
    if 'image' not in request.files:
        print("❌ No image in request")
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    print(f"📄 File received: {file.filename}")
    
    # Check if file is valid
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        print(f"❌ Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        print("💾 Saving file...")
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✅ File saved to: {filepath}")
        
        # Preprocess the image
        print("🔄 Preprocessing image...")
        img = preprocess_image(filepath)
        if img is None:
            print("❌ Could not preprocess image")
            os.remove(filepath)  # Clean up
            return jsonify({'error': 'Could not process image'}), 400
        
        # Make prediction
        print("🤖 Running model prediction...")
        prediction = model.predict(img, verbose=0)[0][0]
        print(f"📊 Raw prediction: {prediction}")
        
        is_potato = bool(prediction >= 0.5)
        confidence = float(prediction) if is_potato else float(1 - prediction)
        
        print(f"✅ Result: {'POTATO' if is_potato else 'NOT POTATO'} (confidence: {confidence:.2f})")
        
        # Clean up uploaded file
        os.remove(filepath)
        print("🗑️ Cleaned up uploaded file")
        
        # Return result
        result = {
            'is_potato': is_potato,
            'confidence': confidence,
            'raw_score': float(prediction),
            'label': '🥔 Potato' if is_potato else '🚫 Not Potato'
        }
        print(f"📤 Sending response: {result}")
        return jsonify(result)
    
    except Exception as e:
        # Clean up on error
        print(f"💥 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🥔 Potato Classifier API Server")
    print("="*50)
    print("📍 Server running on: http://localhost:5000")
    print("📡 Ready to receive requests!")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
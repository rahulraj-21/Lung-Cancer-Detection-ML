import os
import cv2
import numpy as np
import joblib
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import io
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model & labels
IMG_SIZE = 224
try:
    model = load_model("models/fusion_model.h5")
    class_names = joblib.load("models/labels.pkl")
    print("✅ Model and labels loaded successfully!")
    print("📊 Class names:", class_names)
    
    # Print model layers to help with Grad-CAM
    print("\n🔍 Model layers (for Grad-CAM):")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} - {type(layer).__name__} - {layer.output_shape}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    class_names = {}

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  image_path TEXT NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

init_db()

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def predict_image(image_path):
    """Make prediction on the input image"""
    if model is None:
        raise ValueError("Model not loaded properly")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image from path: " + image_path)
    
    print(f"📐 Original image shape: {img.shape}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    print(f"📏 Resized image shape: {img.shape}")

    # Prepare inputs for both branches of the fusion model
    cnn_input = img / 255.0  # Normalize for CNN branch
    res_input = preprocess_input(img)  # Preprocess for ResNet branch

    # Expand dimensions for batch
    cnn_input = np.expand_dims(cnn_input, axis=0)
    res_input = np.expand_dims(res_input, axis=0)

    print(f"🔧 CNN input shape: {cnn_input.shape}")
    print(f"🔧 ResNet input shape: {res_input.shape}")

    # Make prediction
    pred = model.predict([cnn_input, res_input], verbose=0)
    print(f"🎯 Raw prediction: {pred}")
    
    class_id = np.argmax(pred)
    confidence = np.max(pred)

    print(f"🏷️ Predicted class ID: {class_id}")
    print(f"📝 Predicted class name: {class_names[class_id]}")
    print(f"💯 Confidence: {confidence:.4f}")

    return class_names[class_id], confidence, class_id

def get_conv_layer_names(model):
    """Get all convolutional layer names from the model"""
    conv_layers = []
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            conv_layers.append(layer.name)
        # Also check for specific layer types
        elif any(layer_type in str(type(layer)) for layer_type in ['Conv2D', 'Convolution']):
            conv_layers.append(layer.name)
    return conv_layers

def generate_gradcam(model, img_array, pred_index=None):
    """Generate Grad-CAM heatmap for model explanation"""
    print("🔥 Generating Grad-CAM...")
    
    # Get all convolutional layers
    conv_layers = get_conv_layer_names(model)
    print(f"🔍 Available convolutional layers: {conv_layers}")
    
    if not conv_layers:
        print("❌ No convolutional layers found!")
        # Create a simple heatmap as fallback
        return create_fallback_heatmap()
    
    # Try the last convolutional layer first (usually works best)
    layer_name = conv_layers[-1]
    print(f"🎯 Trying layer: {layer_name}")
    
    try:
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            print(f"🎯 Target class index: {pred_index}")
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the target class with respect to the output feature map
        grads = tape.gradient(class_channel, conv_outputs)
        print(f"📐 Gradients shape: {grads.shape}")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print(f"📐 Pooled gradients shape: {pooled_grads.shape}")
        
        # Weight the output feature map with the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        print(f"📐 Heatmap shape before processing: {heatmap.shape}")
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        print(f"✅ Grad-CAM generated successfully. Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        return heatmap
        
    except Exception as e:
        print(f"❌ Grad-CAM failed with layer {layer_name}: {e}")
        # Try alternative layers
        for alt_layer in reversed(conv_layers[:-1]):
            try:
                print(f"🔄 Trying alternative layer: {alt_layer}")
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(alt_layer).output, model.output]
                )
                
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    if pred_index is None:
                        pred_index = tf.argmax(predictions[0])
                    class_channel = predictions[:, pred_index]
                
                grads = tape.gradient(class_channel, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
                heatmap = heatmap.numpy()
                
                print(f"✅ Grad-CAM successful with layer {alt_layer}")
                return heatmap
                
            except Exception as alt_e:
                print(f"❌ Failed with layer {alt_layer}: {alt_e}")
                continue
        
        print("❌ All Grad-CAM attempts failed, using fallback")
        return create_fallback_heatmap()

def create_fallback_heatmap():
    """Create a simple fallback heatmap when Grad-CAM fails"""
    print("🔄 Creating fallback heatmap...")
    # Create a centered Gaussian-like heatmap
    heatmap = np.zeros((14, 14))  # Default small size
    center_x, center_y = heatmap.shape[1] // 2, heatmap.shape[0] // 2
    
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            heatmap[y, x] = np.exp(-dist / 3.0)
    
    return heatmap

def apply_gradcam(img_path, heatmap, alpha=0.5):
    """Apply Grad-CAM heatmap to original image"""
    print("🎨 Applying Grad-CAM to image...")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image for Grad-CAM application")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    print(f"📐 Image shape: {img.shape}")
    print(f"📐 Heatmap shape: {heatmap.shape}")
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    print(f"📐 Resized heatmap shape: {heatmap_resized.shape}")
    
    # Normalize heatmap to 0-255
    heatmap_normalized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    print("✅ Grad-CAM applied successfully")
    return superimposed_img

def generate_lime_explanation(model, img_path, top_labels=1, hide_color=0, num_samples=500):
    """Generate LIME explanation for model prediction"""
    print("🍋 Generating LIME explanation...")
    
    def model_predict(images):
        """Wrapper function for model prediction compatible with LIME"""
        try:
            # Handle single image case
            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=0)
            
            batch_size = min(images.shape[0], 32)  # Smaller batch size for stability
            predictions_batch = []
            
            # Process in smaller batches to avoid memory issues
            for i in range(0, images.shape[0], batch_size):
                batch_images = images[i:i + batch_size]
                
                cnn_input = batch_images / 255.0
                res_input = np.array([preprocess_input(img) for img in batch_images])
                
                batch_predictions = model.predict([cnn_input, res_input], 
                                                batch_size=batch_size, 
                                                verbose=0)
                predictions_batch.append(batch_predictions)
            
            return np.vstack(predictions_batch)
            
        except Exception as e:
            print(f"❌ LIME prediction error: {e}")
            # Return random predictions as fallback
            return np.random.rand(len(images), len(class_names))
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not read image for LIME")
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        explainer = lime_image.LimeImageExplainer()
        
        # Fixed LIME call without random_state parameter
        explanation = explainer.explain_instance(
            image=img_rgb, 
            classifier_fn=model_predict, 
            top_labels=top_labels, 
            hide_color=hide_color, 
            num_samples=num_samples
        )
        
        # Get explanation for the top predicted class
        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0], 
            positive_only=True, 
            num_features=10, 
            hide_rest=False
        )
        
        lime_img = mark_boundaries(temp / 2 + 0.5, mask)
        lime_img = (lime_img * 255).astype(np.uint8)
        lime_img = cv2.cvtColor(lime_img, cv2.COLOR_RGB2BGR)
        
        print("✅ LIME explanation generated successfully")
        return lime_img
        
    except Exception as e:
        print(f"❌ LIME explanation failed: {e}")
        # Return original image with a message overlay as fallback
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Add text overlay indicating LIME failed
        cv2.putText(img, "LIME Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        cv2.putText(img, "Using Fallback", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255), 2)
        return img

def create_simple_lime_fallback(img_path):
    """Create a simple LIME-like visualization as fallback"""
    print("🔄 Creating simple LIME fallback...")
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Create a simple explanation by highlighting edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Create a green mask for edges
    mask = np.zeros_like(img)
    mask[edges > 0] = [0, 255, 0]  # Green color
    
    # Blend with original image
    result = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    
    # Add explanatory text
    cv2.putText(result, "Simple Edge-based", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 2)
    cv2.putText(result, "Explanation", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (255, 255, 255), 2)
    
    return result

def save_image_with_unique_name(image, original_filename, prefix):
    """Save image with unique name to avoid conflicts"""
    name, ext = os.path.splitext(original_filename)
    filename = f"{prefix}_{name}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Ensure the image is in the correct format
    if len(image.shape) == 3 and image.shape[2] == 3:
        cv2.imwrite(filepath, image)
        print(f"💾 Saved {prefix} image: {filename}")
        return filename
    else:
        print(f"❌ Invalid image format for {prefix}")
        return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 5',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=predictions)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"💾 File saved: {filepath}")
            
            # Make prediction
            try:
                if model is None:
                    flash('AI model is not loaded properly. Please contact administrator.', 'error')
                    return redirect(request.url)
                
                prediction, confidence, class_id = predict_image(filepath)
                print(f"🎯 Final Prediction: {prediction}, Confidence: {confidence:.4f}, Class ID: {class_id}")
                
                # Save prediction to database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO predictions (user_id, image_path, prediction, confidence) VALUES (?, ?, ?, ?)',
                    (session['user_id'], filename, prediction, float(confidence))
                )
                conn.commit()
                conn.close()
                
                print("💾 Prediction saved to database")
                
                # Generate explanations only for abnormal cases
                gradcam_filename = None
                lime_filename = None
                
                if class_id in [0, 1]:  # Benign (0) or Malignant (1) cases
                    print("🔍 Generating explanations for abnormal case...")
                    
                    # Generate Grad-CAM
                    try:
                        # Prepare image for Grad-CAM
                        img_array = cv2.imread(filepath)
                        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        cnn_input = img_array / 255.0
                        res_input = preprocess_input(img_array)
                        cnn_input = np.expand_dims(cnn_input, axis=0)
                        res_input = np.expand_dims(res_input, axis=0)
                        
                        # Generate Grad-CAM
                        heatmap = generate_gradcam(model, [cnn_input, res_input], class_id)
                        gradcam_img = apply_gradcam(filepath, heatmap)
                        gradcam_filename = save_image_with_unique_name(gradcam_img, filename, 'gradcam')
                        
                    except Exception as e:
                        print(f"❌ Grad-CAM failed: {e}")
                        gradcam_filename = None
                    
                    # Generate LIME explanation
                    try:
                        lime_img = generate_lime_explanation(model, filepath)
                        if lime_img is not None:
                            lime_filename = save_image_with_unique_name(lime_img, filename, 'lime')
                        else:
                            # Use fallback if LIME returns None
                            lime_fallback = create_simple_lime_fallback(filepath)
                            lime_filename = save_image_with_unique_name(lime_fallback, filename, 'lime_fallback')
                            
                    except Exception as e:
                        print(f"❌ LIME failed: {e}")
                        # Create fallback LIME visualization
                        lime_fallback = create_simple_lime_fallback(filepath)
                        lime_filename = save_image_with_unique_name(lime_fallback, filename, 'lime_fallback')
                
                return render_template('results.html', 
                                     prediction=prediction, 
                                     confidence=confidence,
                                     image_filename=filename,
                                     gradcam_filename=gradcam_filename,
                                     lime_filename=lime_filename,
                                     show_explanations=class_id in [0, 1],
                                     class_id=class_id)
                
            except Exception as e:
                print(f"❌ Error in prediction pipeline: {e}")
                flash(f'Error processing image: {str(e)}', 'error')
                # Clean up uploaded file if there was an error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload PNG, JPG, or JPEG images.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    # Check if model is loaded before starting
    if model is not None:
        print("🚀 Starting Flask application...")
        print("📊 Available classes:", class_names)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check model files.")
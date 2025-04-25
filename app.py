import os
import io
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
from PIL import Image, ImageOps # Added ImageOps for orientation
from werkzeug.utils import secure_filename # Good practice

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure the Gemini API client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as it's generally faster and sufficient for this task
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None # Set model to None if configuration fails

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyzes the uploaded image using Gemini after resizing."""
    if model is None:
         return jsonify({"error": "Gemini API not configured. Check server logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename) # Sanitize filename

    if filename == '':
        return jsonify({"error": "No image selected"}), 400

    if file:
        try:
            # Read image bytes directly from the file stream
            img_bytes = file.read()
            img_size_kb = len(img_bytes) / 1024
            print(f"Original image size: {img_size_kb:.2f} KB")

            # --- Image Resizing & Orientation Fix ---
            img = Image.open(io.BytesIO(img_bytes))

            # Fix orientation based on EXIF data (important for phone photos)
            img = ImageOps.exif_transpose(img)

            # Define max dimensions (adjust as needed)
            # Lowering slightly further for potentially more constrained environments
            max_size = (800, 800)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Gemini SDK can often handle the PIL Image object directly after processing
            processed_img = img # Use the resized PIL object

            # If passing the PIL object fails later, you might need to save to bytes
            # and construct the Part manually (less common now):
            # output_buffer = io.BytesIO()
            # img_format = img.format if img.format in ['JPEG', 'PNG', 'WEBP'] else 'JPEG'
            # img.save(output_buffer, format=img_format, quality=80) # Slightly lower quality ok
            # resized_img_bytes = output_buffer.getvalue()
            # resized_img_size_kb = len(resized_img_bytes) / 1024
            # print(f"Resized image size: {resized_img_size_kb:.2f} KB")
            # import google.ai.generativelanguage as glm
            # processed_img = glm.Part(inline_data=glm.Blob(mime_type=f'image/{img_format.lower()}', data=resized_img_bytes))
            # ---- End of alternative byte method ----

            # Prepare the prompt for Gemini
            prompt = "Identify the main items or objects visible in this image. Provide a list or a short description."

            # Send the processed (resized) image and prompt to Gemini
            response = model.generate_content([prompt, processed_img])

            # Extract the text response
            detected_items = response.text

            return jsonify({"description": detected_items})

        except Image.DecompressionBombError:
            print("Error: Image is too large or could be a decompression bomb.")
            return jsonify({"error": "Image is too large or corrupt. Please use a smaller image."}), 400
        except genai.types.generation_types.StopCandidateException as e:
             print(f"Gemini generation stopped unexpectedly: {e}")
             # Sometimes this happens if the response is empty or flagged internally
             return jsonify({"error": "Analysis failed or content flagged by API."}), 500
        except Exception as e:
            print(f"Error processing image or calling Gemini: {e}")
            import traceback
            traceback.print_exc() # Print detailed error to server logs
            return jsonify({"error": f"An internal error occurred during analysis."}), 500

    return jsonify({"error": "Invalid file uploaded"}), 400

if __name__ == '__main__':
    # Use 0.0.0.0 for Render, debug=False for production
    port = int(os.getenv("PORT", 8080))
    # Set debug based on environment variable, default to False
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

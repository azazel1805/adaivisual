import os
import io
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, Response
from dotenv import load_dotenv
from PIL import Image # Used for image handling

# Load environment variables from .env file (especially for local development)
load_dotenv()

app = Flask(__name__)

# Configure the Gemini API client
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    # Use a model that supports vision, like gemini-1.5-flash or gemini-pro-vision
    # gemini-1.5-flash is often faster and cheaper for general tasks
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # You might want to exit or handle this more gracefully depending on requirements
    model = None # Set model to None if configuration fails

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyzes the uploaded image using Gemini."""
    if model is None:
         return jsonify({"error": "Gemini API not configured. Check API key."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if file:
        try:
            # Read the image file bytes
            img_bytes = file.read()

            # Use PIL to open the image from bytes - ensures it's a valid image format
            # and allows the Gemini library to process it correctly.
            img = Image.open(io.BytesIO(img_bytes))

            # Prepare the prompt for Gemini
            prompt = "Identify the main items or objects visible in this image. Provide a list or a short description."

            # Send the image and prompt to Gemini
            # The SDK handles sending the image data correctly when passed a PIL Image object
            response = model.generate_content([prompt, img])

            # Check for safety ratings or blocks if necessary (optional)
            # if response.prompt_feedback.block_reason:
            #     return jsonify({"error": f"Content blocked: {response.prompt_feedback.block_reason}"}), 400

            # Extract the text response
            detected_items = response.text

            return jsonify({"description": detected_items})

        except genai.types.generation_types.StopCandidateException as e:
             print(f"Gemini generation stopped unexpectedly: {e}")
             # Sometimes this happens if the response is empty or flagged internally
             return jsonify({"error": "Analysis failed or content flagged by API."}), 500
        except Exception as e:
            print(f"Error processing image or calling Gemini: {e}")
            # Log the full error for debugging
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network (and for Render)
    # Debug=True is helpful for development, but turn it OFF for production/deployment
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8080)), debug=False)
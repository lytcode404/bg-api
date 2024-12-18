from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from backgroundremover.bg import remove
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
import os


# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Allow CORS from localhost:3000
CORS(app, origins=["http://localhost:3000"])


def remove_bg(data):
    # Model options: ["u2net", "u2net_human_seg", "u2netp"]
    model_name = "u2net"  # Choose the model as required
    img = remove(data, model_name=model_name,
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=10,
                 alpha_matting_erode_structure_size=10,
                 alpha_matting_base_size=1000)
    return img


@app.route('/remove-bg', methods=['POST'])
def remove_background_api():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    # Get the uploaded image file
    image_file = request.files['image']
    data = image_file.read()

    # Remove the background
    output_img = remove_bg(data)

    # Prepare the image for response
    output = BytesIO(output_img)
    output.seek(0)

    # Send the processed image as a response
    return send_file(output, mimetype="image/png")


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


# Define generation config
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)


@app.route('/generate', methods=['POST'])
def generate():
    """Handle file uploads and interact with the model."""
    file = request.files['file']
    mime_type = file.mimetype
    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    # Upload file to Gemini
    gemini_file = upload_to_gemini(file_path, mime_type=mime_type)

    # Start chat session
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    gemini_file,
                    "give me the best appropriate :\nbg color in hex code, title (short), paragraph (max 2 sentence line) for the uploaded image in json:",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "```json\n[\n  {\n    \"Professionals\": {\n      \"bg_color\": \"#f0f0f0\",\n      \"title\": \"Professional Attire\",\n      \"paragraph\": \"Elevate your work wardrobe with sophisticated pieces. Make a lasting impression with style and professionalism.\"\n    }\n  },\n  {\n    \"Students\": {\n      \"bg_color\": \"#e8f5e9\",\n      \"title\": \"Campus Style\",\n       \"paragraph\": \"Comfort meets style for the perfect school look. Express your personality with trendy and practical outfits.\"\n    }\n  },\n  {\n    \"Travelers\": {\n      \"bg_color\": \"#fce4ec\",\n      \"title\": \"Travel Ready\",\n      \"paragraph\": \"Explore the world in comfort and style. Pack versatile pieces for any adventure.\"\n    }\n  },\n   {\n    \"Casual Fashion\": {\n      \"bg_color\": \"#e3f2fd\",\n      \"title\": \"Everyday Casuals\",\n      \"paragraph\":\"Effortless style for your daily activities. Comfortable and chic looks for any occasion.\"\n    }\n  }\n]",
                    gemini_file,
                ],
            },
        ]
    )

    response = chat_session.send_message(
        "give me the best appropriate bg color in hex code, title (short), paragraph (max 2 sentence line) for the uploaded image in 4json for Professionals, Students, Travelers, Casual Fashion:")

    # Return the response as JSON
    return jsonify({"response": response.text})


if __name__ == '__main__':
    app.run(debug=True)

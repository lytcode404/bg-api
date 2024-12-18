import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize Flask app
app = Flask(__name__)

# Allow CORS from localhost:3000
CORS(app, origins=["http://localhost:3000"])

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
        ]
    )

    response = chat_session.send_message(
        "give me the best appropriate bg color in hex code, title (short), paragraph (max 2 sentence line) for the uploaded image in json:")

    # Return the response as JSON
    return jsonify({"response": response.text})


if __name__ == '__main__':
    # Create the uploads directory if not exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)

from flask import Flask, request, send_file
from flask_cors import CORS
from backgroundremover.bg import remove
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins


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


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO
from gradio_client import Client

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get("prompt", "A beautiful scenery")  # Default prompt

    try:
        # Connect to Gradio Client
        client = Client("black-forest-labs/FLUX.1-schnell")
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )
        file_path = result[0] if isinstance(result, tuple) else result
        with open(file_path, "rb") as f:
            image_data = BytesIO(f.read())
        return send_file(image_data, mimetype="image/webp")
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

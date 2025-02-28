import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.process_video import process_video
from utils.recommendations import recommend 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            return jsonify({"error": "No file uploaded!"})

        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "No selected file!"})

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type! Only MP4, AVI, MOV are     allowed."})
    
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = process_video(filepath)

        # Fetch the final prediction and generate recommendation
        final_prediction = result.get("final_prediction", "Unknown")
        recommendation_text = recommend(final_prediction)  # Get structured recommendation

        # Add recommendation to the JSON response
        result["recommendation"] = recommendation_text

        return jsonify(result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils.keypoints import extract_keypoints
from utils.recommendations import recommend

app = Flask(__name__)

# Define Upload Folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_video():
    """Endpoint to upload a video file, analyze keypoints, and get recommendations."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(file_path)  # Save uploaded video
            keypoints = extract_keypoints(file_path)  # Extract keypoints
            
            recommendations = []
            for frame in keypoints:
                recommendations.append(recommend(frame["gait_class"]))

            return jsonify({
                'message': 'File uploaded and analyzed successfully',
                'filename': unique_filename,
                'file_url': f"/static/uploads/{unique_filename}",
                'keypoints': keypoints,
                'recommendations': recommendations
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format. Only MP4 files are allowed'}), 400

@app.route('/', methods=['GET'])
def home():
    """Render the homepage for video upload."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

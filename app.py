import os
import gradio as gr
import numpy as np
import tensorflow as tf
import joblib
import json
import base64
import cv2
import tempfile
from PIL import Image
from datetime import datetime
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

clf = joblib.load("random_forest_defect_classifier.pkl")
feature_extractor = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')

COMPLAINTS_FILE = "complaints.json"
HTML_TEMPLATE = "templates/complaint_history.html"
CSS_FILE = "static/style.css"

# --- Ensure files exist ---
if not os.path.exists(COMPLAINTS_FILE):
    with open(COMPLAINTS_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(HTML_TEMPLATE):
    with open(HTML_TEMPLATE, "w") as f:
        f.write("<h3>Complaint History</h3>\n<!-- complaint entries will go here -->")

# --- Helper Functions ---
def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"

def predict(img_np):
    img = Image.fromarray(img_np)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    features = feature_extractor.predict(img_array, verbose=0)[0].reshape(1, -1)
    probs = clf.predict_proba(features)[0]

    predicted_label = 1 if probs[1] >= 0.7 else 0
    label_text = "Defective" if predicted_label == 1 else "Non-Defective"

    label_output = gr.update(value=label_text, visible=True)
    description_text_visibility = gr.update(visible=(label_text == "Defective"))
    log_button_visibility = gr.update(visible=(label_text == "Defective"))

    return label_output, img, description_text_visibility, log_button_visibility

def log_complaint(img, description):
    if not description:
        return "Description required to log complaint."

    with open(COMPLAINTS_FILE, "r+") as f:
        data = json.load(f)
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": "Defective",
            "description": description,
            "image": encode_image(img)
        }
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)

    return "Complaint logged successfully."

def view_complaints():
    with open(COMPLAINTS_FILE, "r") as f:
        data = json.load(f)
    if not data:
        return "<p>No complaints logged yet.</p>"

    entries = ""
    for item in reversed(data):
        entries += f"""
        <div class='complaint'>
            <img src='{item['image']}' class='complaint-img'/>
            <div class='complaint-info'>
                <p><strong>Time:</strong> {item['timestamp']}</p>
                <p><strong>Description:</strong> {item['description']}</p>
            </div>
        </div>
        """

    with open(CSS_FILE, "r") as css_file:
        css = f"<style>{css_file.read()}</style>"

    return css + entries

def process_video(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(open(video_file, "rb").read())
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    defective_frames = 0
    total_counted = 0

    frame_skip = max(total_frames // 30, 1)  # Analyze up to 30 frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label_output, _, _, _ = predict(img_rgb)
            if label_output['value'] == "Defective":
                defective_frames += 1
            total_counted += 1

        frame_idx += 1

    cap.release()

    if total_counted == 0:
        return "Unable to classify video."

    defect_ratio = defective_frames / total_counted
    final_label = "Defective" if defect_ratio >= 0.1 else "Non-Defective"

    return final_label

# --- Gradio App ---
with open("static/gradio_theme.css", "r") as f:
    custom_css = f.read()

with gr.Blocks(css=custom_css) as app:

    gr.Markdown("# Railway Track Defect Detection and Complaint Logger")

    with gr.Tab("Defect Detection"):
        task_type = gr.Radio(choices=["Image", "Video"], value="Image", label="Select Detection Type")

        with gr.Group(visible=True) as image_group:
            image_input = gr.Image(label="Upload Track Image")
            detect_button = gr.Button("Check Image", elem_classes="primary-btn")
            label_output = gr.Label(visible=False)
            description_text = gr.Textbox(label="Please describe your comment and log the complaint", visible=False)
            log_button = gr.Button("Log Complaint", elem_classes="secondary-btn", visible=False)
            log_status = gr.Label(visible=False)
            state_image = gr.State()

        with gr.Group(visible=False) as video_group:
            video_input = gr.File(label="Upload Track Video")
            video_detect_button = gr.Button("Check Video", elem_classes="primary-btn")
            video_output = gr.Label(visible=False)

        task_type.change(
            fn=lambda task: (
                gr.update(visible=(task == "Image")),
                gr.update(visible=(task == "Video"))
            ),
            inputs=task_type,
            outputs=[image_group, video_group]
        )

        detect_button.click(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, state_image, description_text, log_button]
        )

        log_button.click(fn=log_complaint, inputs=[state_image, description_text], outputs=log_status)
        log_button.click(fn=lambda: gr.update(visible=True), inputs=[], outputs=[log_status])

        video_detect_button.click(
            fn=lambda: gr.update(visible=True, value="Checking Video... Please wait ⚡"),
            inputs=[],
            outputs=[video_output],
            queue=False
        ).then(
            fn=process_video,
            inputs=video_input,
            outputs=video_output
        )

    with gr.Tab("View Complaints"):
        history_output = gr.HTML()
        view_button = gr.Button("Refresh Complaint History")
        view_button.click(fn=view_complaints, inputs=[], outputs=history_output)

app.launch()

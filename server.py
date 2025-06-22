from flask import Flask, render_template, redirect
import gradio as gr
from app import app as gradio_ui  # import the Blocks interface as 'gradio_ui'
import threading

# ✅ Rename Flask instance to avoid naming conflict
flask_app = Flask(__name__, template_folder="templates")

@flask_app.route("/")
def landing():
    return render_template("upload_landing_page.html")

@flask_app.route("/detect")
def detect():
    return redirect("http://127.0.0.1:7864", code=302)

if __name__ == "__main__":
    # ✅ Start Gradio on a fixed port in a background thread
    threading.Thread(
        target=lambda: gradio_ui.launch(server_name="127.0.0.1", server_port=7864, share=False)
    ).start()

    # ✅ Start Flask server on port 5000
    flask_app.run(port=5000, debug=True)

# Dinesh Added this comment as commit 1
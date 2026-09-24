from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io
from datetime import datetime

app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model("attendance_cnn_model.keras")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Automated Attendance System</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f5f9;
            text-align: center;
            padding: 50px;
        }

        .container {
            width: 500px;
            margin: auto;
            background: white;
            padding: 35px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }

        h1 {
            color: #222;
        }

        input {
            margin: 20px;
            padding: 10px;
        }

        button {
            padding: 12px 25px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #eef6ff;
            border-radius: 10px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Automated Attendance System</h1>

    <p>Upload a student face image</p>

    <form method="POST" enctype="multipart/form-data">

        <input type="file" name="image" accept="image/*" required>

        <br>

        <button type="submit">Mark Attendance</button>

    </form>

    {% if result %}

    <div class="result">

        <h2>Attendance Result</h2>

        <p><b>Student:</b> {{ student }}</p>

        <p><b>Attendance:</b> {{ attendance }}</p>

        <p><b>Confidence:</b> {{ confidence }}%</p>

        <p><b>Date:</b> {{ date }}</p>

        <p><b>Time:</b> {{ time }}</p>

    </div>

    {% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = False
    student = ""
    attendance = ""
    confidence = 0
    date = ""
    time = ""

    if request.method == "POST":

        uploaded_file = request.files["image"]

        if uploaded_file:

            image = Image.open(
                io.BytesIO(uploaded_file.read())
            ).convert("RGB")

            # Resize image to CNN input size
            image = image.resize((50, 50))

            # Convert image to NumPy array
            image_array = np.array(image)

            # Normalize pixel values
            image_array = image_array.astype("float32") / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # CNN prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            predicted_student = encoder.classes_[
                predicted_index
            ]

            confidence = round(
                float(prediction[0][predicted_index]) * 100,
                2
            )

            student = predicted_student
            attendance = "Present"

            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            result = True

    return render_template_string(
        HTML,
        result=result,
        student=student,
        attendance=attendance,
        confidence=confidence,
        date=date,
        time=time
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)

# Загрузка модели с TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1"
model = hub.load(model_url)

def process_image(image):
    # Подготовка изображения для обработки моделью
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = tf.image.resize(image_rgb, (640, 640))
    image_resized = tf.expand_dims(image_resized, axis=0)
    image_resized = tf.image.convert_image_dtype(image_resized, tf.uint8)

    # Получение результатов обнаружения объектов
    results = model(image_resized)

    # Извлечение данных о боксах и классах объектов
    detection_boxes = results["detection_boxes"][0].numpy()
    detection_classes = results["detection_classes"][0].numpy()

    # Отображение результатов на изображении с помощью Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)

    for i in range(len(detection_boxes)):
        confidence = results["detection_scores"][0].numpy()[i]

        ymin, xmin, ymax, xmax = detection_boxes[i]
        class_id = int(detection_classes[i])

        if class_id == 1:
            # Отрисовка ограничивающего прямоугольника
            image_h, image_w, _ = image_rgb.shape
            x, y, w, h = int(xmin * image_w), int(ymin * image_h), int((xmax - xmin) * image_w), int((ymax - ymin) * image_h)
            rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor='green')
            plt.gca().add_patch(rect)
            plt.text(x, y, s=f"Class: {class_id}, confidence: {confidence}", color='green', verticalalignment='top')

    plt.axis('off')
    
    # Сохранение изображения с результатами
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.read()).decode()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            processed_image = process_image(image)
            return render_template('result.html', processed_image=processed_image)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
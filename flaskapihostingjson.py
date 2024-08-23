import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import io
from PIL import Image
import imageio
import openai

openai.api_key = 'pk-nHkcBLngWWWBPDdyVBjJFyOLNItDHDIkLbCeIBuYldOeuLDc'
openai.base_url = "https://api.pawan.krd/cosmosrp/v1/chat/completions"

app = Flask(__name__)

def load_model_trained():
    def AlexNetCE(input_shape = (227, 227, 3), classes = 6):
        X_input = tf.keras.Input(input_shape)
        X = X_input
        X = tf.keras.layers.Conv2D(96, (11, 11), strides = (4, 4), activation = "relu", name = 'conv1')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = tf.keras.layers.Conv2D(256, (5, 5), padding = "same",activation = "relu", name = 'conv2')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv2')(X)
        X = tf.keras.layers.Conv2D(256, (3, 3), padding = "same",activation = "relu", name = 'conv5')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(1))(X)
        X = tf.keras.layers.Dense(4096, activation = "relu", name='fc' + str(2))(X)
        X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)
        
        # Create model
        model = tf.keras.Model(inputs = X_input, outputs = X, name='ALEXNETCE')

        return model
    model = AlexNetCE()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005), 
                loss='categorical_crossentropy', metrics=['accuracy'])
    #latest = tf.train.latest_checkpoint(os.path.join("training_1", "cp.weights.h5"))
    model.load_weights(filepath="training_1/cp.weights.h5")

    model.summary()
    return model
model = load_model_trained()
@app.route('/predict', methods=['POST'])
def classify_image():
    data = request.get_json()
    if 'file' not in data:
        return jsonify({"error": "No file provided"}), 400
    
    # Decode base64 to bytes, which is equivalent to Uint8List
    image_data = base64.b64decode(data['file'])
    location = data['location']
    # Read the image file
    
    # Convert bytes to image
    img = Image.open(io.BytesIO(image_data))
    
    # Resize and preprocess the image
    img_resized = img.resize((227, 227))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)
    prediction = model.predict(img_array)
    classes = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"How do I recycle {classes[np.argmax(prediction)]} in the {location}? Respond in a short and concise paragraph."},
        ],
    )
    print(completion.choices[0].message.content)

    # print(prediction)
    # print(classes[np.argmax(prediction)])

    return jsonify({'class': classes[np.argmax(prediction)], 'information': completion.choices[0].message.content})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500)
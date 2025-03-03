import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_name = validation_set.class_names
print(class_name)

cnn = tf.keras.models.load_model('trained_model.keras')


import cv2
import matplotlib.pyplot as plt

image_path = 'Test/nevus/ISIC_0000009.jpg'

# Read the image
img = cv2.imread(image_path)

# Check if the image was loaded
if img is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(img)
    plt.title('Test Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()



# Load and preprocess the image
image = tf.keras.utils.load_img(image_path, target_size=(128, 128))  # ✅ Corrected function
input_arr = tf.keras.utils.img_to_array(image)  # ✅ Corrected function
input_arr = np.expand_dims(input_arr, axis=0)  # ✅ Ensures correct batch format

# Print shape
print(input_arr.shape)  # Should be (1, 128, 128, 3)


prediction = cnn.predict(input_arr)  # ✅ Corrected `.predict()`
print(prediction.shape,prediction)  # ✅ Check the output shape



result_index = np.argmax(prediction)
result_index

class_name = ['actinic keratosis',
 'basal cell carcinoma',
 'dermatofibroma',
 'melanoma',
 'nevus',
 'pigmented benign keratosis',
 'seborrheic keratosis',
 'squamous cell carcinoma',
 'vascular lesion']


model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f'Disease-name-{model_prediction}')
plt.xticks([])
plt.yticks([])
plt.show()


model_prediction
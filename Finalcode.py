from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing  import image
import numpy as np
model = load_model (r"E:\VIT\PlantFertilizerRecommendation\Flaskweb\fruitnames.h5")
img = image.load_img("Peach___Bacterial_spot.jpg",target_size = (64,64))
print(type(img))
x = image.img_to_array(img)
print(x.shape)
print(type(x))
x = np.expand_dims(x,axis = 0)
x.shape
pred = np.argmax(model.predict(x))
print(pred)
index = ['Apple___Black_rot','Apple___healthy', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Peach___Bacterial_spot','Peach___healthy']
prediction = index[pred]
print(prediction)
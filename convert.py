import keras

model = keras.models.load_model("my_model.h5")

model.export("my_model/")
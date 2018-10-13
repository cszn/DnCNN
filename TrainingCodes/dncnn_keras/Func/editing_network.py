import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json
from keras.models import load_model
from self_defined_layer import AdLayer
import os

class NetworkEdit_base(object):

	def __init__(self, model=None, opt=None):
		self.model = model
		self.opt = opt


	def TransModel(self):
		new_model = Sequential()
		new_model.add(AdLayer(input_shape = self.model.input_shape[1:]))
		for i in range(len(self.model.layers)):
			new_model.add(self.model.layers[i])
		self.trans_model = new_model


	def SaveModel(self, save_content="MODEL", save_path="./", save_filename="model.h5"):
		assert self.model, "MODEL cannot be NULL"
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
		model_path = os.path.join(save_path, save_filename)
		if save_content == "MODEL":
			self.model.save(model_path)
			print('Save model at %s ' % model_path)

		if save_content == "ARCHITECTURE":
			model_json = self.model.to_json()
			with open(model_path,"w") as json_file:
				json_file.write(model_json)
				json_file.close()

		if save_content == "WEIGHTS":
			self.model.save_weights(model_path)

		assert save_content in ["MODEL",  "ARCHITECTURE", "WEIGHTS"], "save_content needs to be MODEL or ARCHITECTURE or WEIGHTS."

	def LoadModel(self, load_content, load_path):
		assert os.path.exists(load_path), "File does not exist, check the load in path"

		if load_content == "WEIGHTS":
			assert self.model, "MODEL cannot be NULL"
			self.model.load_weights(load_path, by_name=True)

		if load_content == "ARCHITECTURE":
			json_file = open(load_path, 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.model = model_from_json(loaded_model_json)

		if load_content == "MODEL":
			self.model = load_model(load_path)

		assert load_content in ["MODEL",  "ARCHITECTURE", "WEIGHTS"], "save_content needs to be MODEL or ARCHITECTURE or WEIGHTS."

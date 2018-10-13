import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from ResNet_github import ResnetBuilder

class NetworkBuild_base(object):
	#
	# for specific task, build network with required architecture
	# Several parameters needed to be set by user
	#	task 		 --> two main category, "classification" and "regression"
	#	input_shape  --> the required shape fed into the network
	#	output_shape --> the required shape get out of the network
	#	pre_train	 --> flag, when we load in a typical model, needed pre-trained weights or random weights
	#	net_name	 --> the typical architecture we need, like VGG, ResNet
	#	customized_model --> architecture designed by user can also be fed into the class.
	#

	def __init__(self, task, input_shape, output_shape, pre_train, net_name, customized_model=None):
		self.task = task
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.pre_train = pre_train
		self.net_name = net_name
		self.model = customized_model
		if self.model == None:
			self.BuildNet()

	def BuildNet(self):
		if self.task == "classification":
			if self.pre_train:
				self.LoadModel()
			else:
				self.BuildModel()


		if self.task == "regression":
			pass

	def BuildModel(self):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=self.input_shape))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(self.output_shape))
		model.add(Activation('softmax'))
		self.model = model


	def LoadModel(self):
		if self.net_name == "VGG16":
			from keras.applications.vgg16 import VGG16
			self.model = VGG16(weights='imagenet', include_top=False,
				input_shape=self.input_shape,classes=self.output_shape)

		if self.net_name == "VGG19":
			from keras.applications.vgg19 import VGG19
			self.model = VGG19(weights='imagenet', include_top=False,
				input_shape=self.input_shape,classes=self.output_shape)

		if self.net_name == "ResNet50":
			from keras.applications.resnet50 import ResNet50
			self.model = ResNet50(weights='imagenet', include_top=False,
				input_shape=self.input_shape,classes=self.output_shape)

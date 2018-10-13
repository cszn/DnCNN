import keras
import os

class NetworkRun_base(object):

	def __init__(self, model, opt = keras.optimizers.SGD(), 
							  loss = keras.losses.mean_squared_error,
							  input_train=None, output_train=None, batch_size=1,epochs=1):
# 							  save_dir=None, model_name="default.h5"):
		self.model = model
		self.opt = opt
		self.loss = loss
		self.input_train = input_train
		self.output_train = output_train
		self.batch_size = batch_size
		self.epochs = epochs
# 		self.save_dir = save_dir
# 		self.model_name = model_name
		self.TrainModel()
# 		self.SaveModel()

	def TrainModel(self):
		self.model.compile(optimizer=self.opt,
						   loss=self.loss)
		self.model.fit(x=self.input_train,
					   y=self.output_train,
					   batch_size=self.batch_size,
					   epochs=self.epochs,
					   verbose=1)


	## model save function has been moved to another headfile, named as editing_network.py
	# def SaveModel(self):
	# 	if not os.path.isdir(self.save_dir):
	# 		os.makedirs(self.save_dir)
	# 	model_path = os.path.join(self.save_dir, self.model_name)
	# 	self.model.save(model_path)
	# 	print('Saved trained model at %s ' % model_path)


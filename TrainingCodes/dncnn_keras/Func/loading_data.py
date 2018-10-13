import os




script_name = 'loading_data.py'
dir_path = os.path.dirname(os.path.realpath(script_name))
parent_path = os.path.dirname(dir_path)
loaddata_path = os.path.join(parent_path,'data')

class DataLoader_base(object):
# loading data from external/internal file, chaging the data format into standard format, easy to feed 
# into other modules, like network/data augumentation
# There are several parameters need to be fed into the class:
#		data_source  --> where the data comes from, 
#						 "inside" means from some predefine function/module, currently using mnist and cifar10 
#						 "outside" means from outside folders
#		filename      --> the name of the dataset, 
#						  when data_source="outside", file path needs to be included
#		data_validate --> whether to validate the shape of trainng and dataset, see if they are match
#

	def __init__(self, data_source="inside",filename='mnist',data_validate=True):
		self.data_source = data_source
		self.filename = filename
		self.data_validate = data_validate

		self.GetData()
		self.DataValidation()
		
		

	def GetData(self):
		from keras.datasets import mnist
		from keras.datasets import cifar10

		if self.data_source == "inside":
			if self.filename == "mnist":
				(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
				X_train = X_train.reshape(X_train.shape[0], 28,28, 1)
				X_test = X_test.reshape(X_test.shape[0], 28,28, 1)
			if self.filename == "cifar10":
				(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

		self.input_train = X_train
		self.output_train = Y_train
		self.input_test = X_test
		self.output_test = Y_test

	def DataValidation(self):
		if self.data_validate:
			assert self.input_train.shape[0] == self.output_train.shape[0]
			assert self.input_test.shape[0] == self.output_test.shape[0]
			assert self.input_train.shape[1:] == self.input_test.shape[1:]
			assert self.output_train.shape[1:] == self.output_test.shape[1:]
			info = {}
			info["dataset_name"] = self.filename
			info["validation_status"] = self.data_validate
			info["num_train"] = self.input_train.shape[0]
			info["num_test"] = self.input_test.shape[0]
			info["input_shape"] = self.input_train.shape[1:]
			info["output_shape"] = self.output_train.shape[1:]
			self.info = info 
			    
    
    


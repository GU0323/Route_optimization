# import the necessary packages
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers.normalization import BatchNormalization
# from tensorflow.keras.layers.convolutional import Conv2D
# from tensorflow.keras.layers.convolutional import MaxPooling2D
# from tensorflow.keras.layers.core import Activation
# from tensorflow.keras.layers.core import Dropout
# from tensorflow.keras.layers.core import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def create_mlp_2layer(dim, _node1, _node2, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(_node1, input_dim=dim, activation="relu"))
	model.add(Dense(_node2, activation="relu"))
	model.add(Dropout(0.3))
	# model.add(Dense(16, input_dim=dim, activation=LeakyReLU()))
	# model.add(Dense(4, activation=LeakyReLU()))
	# alpha = 0.1

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# model = Sequential()
	# # model.add(Dense(1, input_dim=dim, activation="linear", use_bias=False))
	# model.add(Dense(1, input_dim=dim, activation="linear"))

	# return our model
	return model

def create_mlp_3layer(dim, _node1, _node2, _node3, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(_node1, input_dim=dim, activation="relu"))
	model.add(Dense(_node2, activation="relu"))
	model.add(Dense(_node3, activation="relu"))
	#model.add(Dropout(0.2))
	# model.add(Dense(16, input_dim=dim, activation=LeakyReLU()))
	# model.add(Dense(4, activation=LeakyReLU()))
	# alpha = 0.1

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="softmax"))

	# model = Sequential()
	# # model.add(Dense(1, input_dim=dim, activation="linear", use_bias=False))
	# model.add(Dense(1, input_dim=dim, activation="linear"))

	# return our model
	return model
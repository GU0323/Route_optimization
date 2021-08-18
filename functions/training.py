# import the necessary packages
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from functions import datasets
from functions import models
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def traning_Pdata(_numberOfIteration, _inputPath, _featureList, _nodeList, _model_description, _testDataRate=0.1):
	mean_list = []
	Rsquared_list = []
	RMSE_list = []
	loss_list = []

	for k in range(_numberOfIteration):

		#모델 저장용 디랙토리 생성
		directory = "Models/{}-{}".format(_model_description, int(k))
		if not os.path.exists(directory):
			os.makedirs(directory)

		# construct the path to the input .txt file that contains information
		# on each house in the dataset and then load the dataset
		print("[INFO] loading welding attributes...")
		# inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])

		FOC_df = datasets.load_FOC_data(_inputPath)

		# construct a training and testing split with 75% of the data used
		# for training and the remaining 25% for evaluation
		# print("[INFO] constructing training/testing split...")
		if (_testDataRate < 0.000001):
			train = FOC_df
			test = FOC_df
		else:
			(train, test) = train_test_split(FOC_df, test_size=_testDataRate)



		#maxNI = P_df["Consumption"].max()
		trainY = train["Consumption"]
		testY = test["Consumption"]
		trainY = np.hstack([trainY])
		testY = np.hstack([testY])


		print("[INFO] processing data...")

		trainX = datasets.process_consumption(True, train, _featureList, directory)
		testX = datasets.process_consumption(False, test, _featureList, directory)


		if (len(_nodeList) == 2):
			model = models.create_mlp_2layer(trainX.shape[1], _nodeList[0], _nodeList[1], regress=True)
		if (len(_nodeList) == 3):
			model = models.create_mlp_3layer(trainX.shape[1], _nodeList[0], _nodeList[1], _nodeList[2], regress=True)

		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=1e-3,
			decay_steps=100,
			decay_rate=0.99)

		#optimizer = Adam(learning_rate=lr_schedule, decay=1e-3 / 200)
		optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-06)
		model.compile(loss="mean_squared_error", optimizer=optimizer)
		history = model.fit(trainX, trainY, epochs=1000, batch_size=10)

		'''

        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss="mean_squared_error", optimizer=opt)



        def lr_scheduler(epoch, lr):
            if epoch > 1200:
                lr = 0.000000000001
                return lr
            if epoch > 900:
                lr = 0.0000000001
                return lr
            if epoch > 600:
                lr = 0.00000001
                return lr
            if epoch > 300:
                lr = 0.000001
                return lr
            return lr
        callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]

        model.summary()

        # train the model
        print("[INFO] training model...")
        # model.fit(P_trainX, P_trainY, validation_data=(testX, testY),epochs=200, batch_size=8)
        model.fit(P_trainX, P_trainY, epochs=1500, batch_size=4, callbacks=callbacks)
        '''


		# 학습된 모델 저장
		#modelFile = "/Models/{}/model-{}-{}".format(_model_description, _model_description, int(k))
		modelFile = "./{}".format(directory)
		model.save(modelFile)

		# 학습한 모델을 이용한 예측 테스트
		print("[INFO] predicting Consumption...")
		preds = model.predict(testX)

		predicted = preds.flatten()
		observed = testY
		mean, Rsquared = datasets.CalcRsquared(predicted, observed)
		RMSE = datasets.CalcRMSE(predicted, observed)

		result = np.array([testY, preds.flatten()])
		result = result.transpose()
		Labels = ['Label', 'Predicted']
		result_df = pd.DataFrame.from_records(result, columns=Labels)

		# 예측 테스트 저장
		directory = "Results/{}".format(_model_description)
		if (Rsquared > 0 and Rsquared < 1):
			if not os.path.exists(directory):
				os.makedirs(directory)
			export_path = "Results/{}/result-{}-{}.csv".format(_model_description, _model_description, int(k))
			# result_df.to_excel(export_path, sheet_name='Sheet1')
			result_df.to_csv(export_path)

			#loss_list.append(history.history['loss'])
			#print(loss_list)
			mean_list.append(mean)
			Rsquared_list.append(Rsquared)
			RMSE_list.append(RMSE)


			#plt.plot(history.history['loss'], 'r-')
			#plt.rc('font', size=12)
			#plt.xlabel('epochs')
			#plt.ylabel('loss')
			#plt.ylim([0, 1.5])
			#plt.ylabel('val_loss')
			#plt.show()

	a = len(mean_list)
	f = open('Results/{}/result-{}-Summary.txt'.format(_model_description, _model_description), mode='wt', encoding='utf-8')
	for i in range(len(mean_list)):
		f.writelines("[INFO] mean: {:.2f} Rsquared: {:.2f} RMSE: {:.2f} \n".format(mean_list[i], Rsquared_list[i], RMSE_list[i]))

	av_mean = np.average(mean_list)
	av_Rsquared = np.average(Rsquared_list)
	max_Rsquared = np.max(Rsquared_list)
	av_RMSE = np.average(RMSE_list)
	max_RMSE = np.max(RMSE_list)
	f.writelines("[Summary] mean: {:.2f} Rsquared_MEAN: {:.2f} Rsquared_MAX: {:.2f} RMSE_MEAN: {:.2f} RMSE_MAX: {:.2f}\n".format(av_mean, av_Rsquared, max_Rsquared, av_RMSE, max_RMSE))
	#f.writelines("[Summary] loss: {}\n".format(loss_list))


	f.close()
# tensorflow, keras, scikit
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras import backend as K
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn.model_selection as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# helper imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MLChallengeHelpers:

	# hyperparamter gird space options
	optimizer = ['Adam', 'SGD']
	activation = ['relu', 'sigmoid']
	batch_size = [50, 100] 
	epochs = [5, 10]
	neurons = [64,128,284]

	# hyperparamter gird space
	hyperparamter_space = dict(
	    batch_size=batch_size, 
	    epochs=epochs,
	    optimizer=optimizer,
	    activation=activation,
	    neurons=neurons
	)
    
	# f1_score calculation function for model evaluation
	def f1_score(y_true, y_pred):

	    # Count positive samples.
	    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

	    # If there are no true samples, fix the F1 score at 0.
	    if c3 == 0:
	        return 0

	    # How many selected items are relevant?
	    precision = c1 / c2

	    # How many relevant items are selected?
	    recall = c1 / c3

	    # Calculate f1_score
	    f1_score = 2 * (precision * recall) / (precision + recall)
	    return f1_score
    
	# reading csv    
	def read_csv(filepath, has_header=None):
		# data import
		data = pd.read_csv(filepath, header=has_header).values

		return data

	# model creation functoin
	def create_model(f1_score=f1_score,neurons=128,activation='relu',optimizer='sgd'):

	    # model init; sequential MLP 
	    model = Sequential()
        
	    # layer1: input layer, with kernel initiazer function same as class distriubtion function        
	    model.add(
	    	Dense(256, activation=activation, input_shape=(284,), kernel_initializer = 'normal')
	    )
        
	    # layer2: dropout for preventing overfitting 
	    model.add(Dropout(0.5))
        
	    # layer3: activation        
	    model.add(
	    	Dense(128, activation=activation)
	    )

	    # layer4: dropout
	    model.add(Dropout(0.2))        
        
	    # layer5: output with softmax        
	    model.add(
	    	Dense(5, activation=tf.nn.softmax)
	    )
	    
	    # print model summary 
	    print(model.summary())

	    # compile model
	    model.compile(
	        optimizer=optimizer, 
	        loss='sparse_categorical_crossentropy',
	        metrics=[f1_score, 'accuracy'] # f1 score
	    )

	    return model

	# fit model
	def fit_model(model, x_train, y_train, epochs=10, verbose=1):

		# fit model and retrieve history 
		history = model.fit(x_train, y_train, epochs=epochs,verbose=verbose)

		# Plot training & validation f1 values
		plt.plot(history.history['f1_score'])
		plt.title('Model F1')
		plt.ylabel('F1')
		plt.xlabel('Epoch')
		plt.legend(['Train'], loc='upper left')
		plt.show()

		return history

	# evaluate model
	def evaluate_model(model, x_test, y_test):

		test_loss, test_f1, test_acc = model.evaluate(x_test, y_test)

		print('Test F1 Score:', test_f1)

		return test_loss, test_f1, test_acc

	# hyperparamter optimization using scikit learn GridSearch CV
	def optimize_hyperparamters(build_fn, x_train, y_train, hyperparamter_space=dict()):
        
		# init classifier
		model = KerasClassifier(
			build_fn=build_fn, verbose=0
		)

		# gridsearch using hyperparamer space        
		grid = sk.GridSearchCV(
			estimator=model, 
			param_grid=hyperparamter_space, 
			cv=3,
			n_jobs=-1
		)
		
		# model fitting        
		grid_result = grid.fit(
			x_train, 
			y_train
		)
		# print best fit 
		print("Best Params: %s; Score: %f" % (grid_result.best_params_, grid_result.best_score_))

		means = grid_result.cv_results_['mean_test_score']
		params = grid_result.cv_results_['params']
        
		# print all combinations
		for mean, param in zip(means, params):
			print("%f @ %r" % (mean, param))

		return grid_result.best_params_

	# model evaluation functions
	def model_evaluate(model, x_train, y_train, x_test, y_test):
		
		# get training history        
		history = model.fit(x_train, y_train, epochs=10,verbose=1)
        
		# evaluate model on test        
		test_loss, test_f1, test_acc = model.evaluate(x_test, y_test)

		# Plot training & validation f1 values
		plt.plot(history.history['f1_score'])
# 		plt.plot(history.history['val_f1_score'])
		plt.title('Model F1')
		plt.ylabel('F1')
		plt.xlabel('Epoch')
		plt.legend(['Train'], loc='upper left')
		plt.show()
        
		return history,test_loss, test_f1, test_acc

	# split for cross validation     
	def split_by_test_size(x, y, test_size=0.33):

		# split 1: train, test
		x_train, x_test, y_train, y_test = sk.train_test_split(
		    x,
		    y,
		    test_size=test_size
		)

		return x_train, x_test, y_train, y_test
    
	# normalize data by min max     
	def normalize_matrix_by_min_max(matrix):
        
# 		return (
#             MinMaxScaler(
#                 copy=True, feature_range=(0, 1)
#             ).fit_transform(matrix)
#         )
		sum_vector = (matrix - np.min(matrix))/np.ptp(matrix)
		return sum_vector    

	# encode class labels     
	def encode_vector_class_to_int(vector_class):
		
		# change str labels to int
		classes_list, encoded = np.unique(vector_class, return_inverse=True)
        
		# # one hot encoding
		# encoded = np_utils.to_categorical(encoded)
    
		return classes_list, encoded

	# drop all zero column vectors     
	def drop_all_zero_vectors(raw_matrix, print_zero_columns_indices=False):
		all_zero_columns_mask= np.sum(raw_matrix,axis=0) > 0
		all_zero_columns_indices=np.where(all_zero_columns_mask==False)        
		all_zero_columns=np.delete(raw_matrix, all_zero_columns_indices, axis=1)
        
		if print_zero_columns_indices:
			print(all_zero_columns_indices)
        
		return all_zero_columns 

	# class distribution of matrix
	def get_class_distribution(vector, draw_histogram=True, xticks = [], xlabels=[]):
		plt.xticks(xticks, xlabels)

        # class distribution, C over represeted, A and E under represented
		if draw_histogram:
			sns.distplot(vector, kde=False, rug=True)

        # bins' size
		class_dist = np.bincount(labels)

		return class_dist
    
	def get_top_n_correlated(features, labels, n=10, feature_indices=[]):

		x=features
		y=labels

		# change str labels to int         
		u, y = np.unique(y, return_inverse=True)

		# get features if available 
		if feature_indices:
			x = x[:, feature_indices]

		# reshape labels from vector to matrix     
		y = y[:].reshape(len(labels),1)

		# merge features and label matrices. shape: (66137, 295)          
		z = np.hstack([
			x.astype(float),
			y.astype(float)
		])

		# convert numpy arr to pandas datafram          
		df = pd.DataFrame(z)

		# calculate pearson's correlation coefficient matrix         
		corr = df.corr() # df is the pandas dataframe

		# plot pearson's correlation matrix on heatmap 
		plt.figure(figsize=(15,15))        
		sns.heatmap(corr)
		plt.show()

		# unstack correlation matrix 
		c1 = corr.abs().unstack()

		# get pairs with strongest correlation skipping columns' own pair (diagonal where correlation is 1)         
		top_n_correlated = c1.sort_values(ascending = False)

		# skip columns' own pair (diagonal where correlation is 1)   
		top_n_correlated = top_n_correlated.iloc[features.shape[1]+1:]

		# plt.show()

		return top_n_correlated.iloc[:n+1]
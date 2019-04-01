
from MLChallenge import MLChallengeHelpers as ml

# read csv
data = ml.read_csv('sample.csv')

# slice feature, label
features = data[:,:-1]
labels = data[:,-1]

# get top 10 correlated vectors from matrix
top_10_correlated = ml.get_top_n_correlated(features, labels, n=10)
          
# get heatmap of top 10 correlated vectors
top_10_correlated = ml.get_top_n_correlated(features, labels, n=10, feature_indices=[127,71,140,72,80,278,1,195,11,6,43,3])

# drop All-zero vector
features = ml.drop_all_zero_vectors(features)

# # normalize features
features = ml.normalize_matrix_by_min_max(features)

# # encode label
labels_list, labels = ml.encode_vector_class_to_int(labels)

# split train, test
x_train, x_test, y_train, y_test = ml.split_by_test_size(
	features, 
	labels
)

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

# # get optimized hyperparamter combination
# best_params = ml.optimize_hyperparamters(
# 	ml.create_model, 
# 	x_train, 
# 	y_train, 
# 	hyperparamter_space
# )

# create model
model = ml.create_model()

# model fitting
history = ml.fit_model(model, x_train, y_train, epochs=10,verbose=1)

# model evaluation
loss, f1_score, accuracy = ml.evaluate_model(model, x_test, y_test)

from pydephasing.log import log
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from tensorflow import keras
from pydephasing.mpi import mpi
#
# here we define the neural network class
# two concrete subclasses :
# 1) multilayer perceptron
# 2) keras DNN -> from pytorch

#
# function to create the NN
# instance
def generate_NN_object(NN_model):
    if NN_model == "MLP":
        return MLP_model_class()
    elif NN_model == "DNN":
        return DNN_model_class()
    else:
        log.error("Wrong neural network model selection : MLP / DNN")
#
#  neural network base class
class NN_model_base_class:
    def __init__(self):
        self.NN_model = None
        self.regr = None
#
# concrete MLP class
class MLP_model_class(NN_model_base_class):
    def __init__(self):
        super(MLP_model_class, self).__init__()
    def set_model(self, NN_parameters):
        n_hidden_layers = tuple(NN_parameters['n_hidden_layers'])
        activation = NN_parameters['activation']
        solver = NN_parameters['solver']
        alpha = NN_parameters['alpha']
        max_iter = NN_parameters['max_iter']
        random_state = NN_parameters['random_state']
        # define NN model
        self.NN_model = MLPRegressor(hidden_layer_sizes=n_hidden_layers,
                        activation=activation, solver=solver,
                        alpha=alpha, random_state=random_state, 
                        max_iter=max_iter)
    # fit NN model
    def fit(self, NN_parameters, X, y):
        random_state = NN_parameters['random_state']
        test_size = NN_parameters['test_size']
        shuffle = NN_parameters['shuffle']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size, shuffle=shuffle)
        # build regressor
        self.regr = self.NN_model.fit(X_train, y_train)
        return X_test, y_test
    # get the score
    def get_score(self, X_test, y_test):
        if mpi.rank == mpi.root:
            log.info("\t N. LAYERS MULTILAYER PERCEPTRON MODEL: " + str(self.regr.n_layers_))
            log.info("\t MODEL SHAPE: " + str(len(self.regr.coefs_)))
        score = self.regr.score(X_test, y_test)
        return str(score)
    # predict value
    def predict(self, X):
        y = self.regr.predict(X)
        return y
#
# concrete DL model class
class DNN_model_class(NN_model_base_class):
    def __init__(self):
        super(DNN_model_class, self).__init__()
    def set_model(self, NN_parameters):
        n_hidden_layers = NN_parameters['n_hidden_layers']
        n_hidden_units  = NN_parameters['n_hidden_units']
        activation_in = NN_parameters['activation_in']
        activation_hid= NN_parameters['activation_hid']
        activation_out= NN_parameters['activation_out']
        loss = NN_parameters['loss']
        optimizer = NN_parameters['optimizer']
        # build the model
        self.NN_model = keras.Sequential()
        self.NN_model.add(keras.layers.Dense(units=1, activation=activation_in, input_shape=[1]))
        for n in range(n_hidden_layers):
            self.NN_model.add(keras.layers.Dense(units=n_hidden_units,activation=activation_hid))
        self.NN_model.add(keras.layers.Dense(units=1, activation=activation_out))
        # compile model
        self.NN_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        # display model
        info = str(self.NN_model.summary())
        if mpi.rank == mpi.root:
            log.info(info)
    # fit NN model
    def fit(self, NN_parameters, X, y):
        epochs = NN_parameters['epochs']
        verbose= NN_parameters['verbose']
        # fitting
        self.NN_model.fit(X, y, epochs=epochs, verbose=verbose)
    # get score
    def get_score(self, X_test, y_test):
        score, acc = self.NN_model.evaluate(X_test, y_test)
        if mpi.rank == mpi.root:
            log.info("NN model accuracy level : " + str(acc))
        return str(score)
    # predict value
    def predict(self, X):
        y = self.NN_model.predict(X)
        return y
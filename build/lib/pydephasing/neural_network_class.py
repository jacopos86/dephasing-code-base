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
        
#
# concrete MLP class
class MLP_model_class(NN_model_base_class):
    def __init__(self):
        super(MLP_model_class, self).__init__()
        
#
# concrete DL model class
class DNN_model_class(NN_model_base_class):
    def __init__(self):
        super(DNN_model_class, self).__init__()
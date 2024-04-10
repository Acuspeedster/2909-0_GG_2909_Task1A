
####################### IMPORT MODULES #######################
import pandas as pd
import torch
import numpy as np


################# ADD UTILITY FUNCTIONS HERE #################
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
#from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




##############################################################

def data_preprocessing(task_1a_dataframe):

	''' 
	Purpose:
	---
	This function will be used to load the csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						  Pandas dataframe read from the provided dataset 	
	
	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to 
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
	encoded_dataframe = task_1a_dataframe.copy()
    
    # Encode categorical variable
	le = LabelEncoder()
	encoded_dataframe['Gender'] = le.fit_transform(encoded_dataframe['Gender'])
	encoded_dataframe['EverBenched'] = le.fit_transform(encoded_dataframe['EverBenched'])
	encoded_dataframe['Education'] =  le.fit_transform(encoded_dataframe['Education'])
	encoded_dataframe['City'] =  le.fit_transform(encoded_dataframe['City'])
	encoded_dataframe['JoiningYear'] =  le.fit_transform(encoded_dataframe['JoiningYear'])

	
	return encoded_dataframe
	

def identify_features_and_targets(encoded_dataframe):
	'''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	
	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''

	
	x = encoded_dataframe.iloc[:,:-1]
	y = encoded_dataframe.iloc[:,[-1]]
	features_and_targets=[x,y]
	

	return features_and_targets


def load_as_tensors(features_and_targets):
    '''
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to an iterable dataset
    object.
    
    Input Arguments:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the 
                            selected features and the second item is the target label
    
    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch tensor
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as Pytorch tensor in validation data
                                            [4]: Iterable dataset object for training data

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''

   
    global X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    X = features_and_targets[0].select_dtypes(include=[np.number]).values
    y = features_and_targets[1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]
    
    
    return tensors_and_iterable_training_data


class Salary_Predictor(nn.Module):
	'''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		'''
		Define the type and number of layers
		'''
		
		self.fc1 = nn.Linear(8, 128)
		self.relu = nn.ReLU()
		# self.fc2 = nn.Linear(128, 64)
		# self.relu = nn.ReLU()
		self.fc3 = nn.Linear(128, 32)
		self.relu = nn.ReLU()
		self.fc4 = nn.Linear(32, 1)
		self.sigmoid = nn.Sigmoid()
		

	def forward(self, x):
		'''
		Define the activation functions
		'''
		
		out = self.fc1(x)
		out = self.relu(out)
		# out = self.fc2(out)
		# out = self.relu(out)
		out = self.fc3(out)
		out = self.relu(out)
		out = self.fc4(out)
		out = self.sigmoid(out)

		predicted_output = out
		

		return predicted_output

def model_loss_function():
	'''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.
    
    Input Arguments:
    ---
    None
    
    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    
	
    # Defined the loss function here
	loss_function = nn.BCEWithLogitsLoss()

    	

	
	return loss_function

def model_optimizer(model):
	'''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
	
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	

	return optimizer

def model_number_of_epochs():
	'''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
	
	number_of_epochs = 20
	

	return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and an iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''    
    
    train_loader = tensors_and_iterable_training_data[4]

    for epoch in range(number_of_epochs):
        print(epoch)
        for batch in train_loader:
            # Extract batch data and labels
            X_data, y_data = batch

            # Forward pass
            y_pred = model(X_data)

            # Modify the shape of y_data to match y_pred
            y_data = y_data.view(-1, 1)  # Reshape y_data to [16, 1]

            # Calculate loss
            ls = loss_function(y_pred, y_data)

            # Backpropagation and optimization
            ls.backward()
            optimizer.step()
            optimizer.zero_grad()


    trained_model = model
   

    return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilize the trained model to make predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and an iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)
    '''

    
    y_test=tensors_and_iterable_training_data[3]
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    y_pred = (y_pred >= 0.5).to(torch.float32)
    model_accuracy = accuracy_score(y_test_tensor, y_pred)
    precision = precision_score(y_test_tensor, y_pred)
    recall = recall_score(y_test_tensor, y_pred)
    f1 = f1_score(y_test_tensor, y_pred)
    

    return model_accuracy, precision, recall, f1
    

'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''

if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pd.read_csv('task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0]
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")
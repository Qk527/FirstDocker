#Train module to Docker test

#Paths
path = "data_floder"
bin_path = "bins_folder"
#Libraries

#General purpose
import pandas as pd
#Preprocessing
from sklearn import preprocessing
#Model
from sklearn.neural_network import MLPClassifier
#Metrics
from sklearn.metrics import classification_report
#Export
from joblib import dump

def train():

    # Load, read and normalize training data
    data_train = pd.read_csv(path+"\\train.csv")

    #Drop not necessary feature
    data_train.drop('Line',axis=1,inplace=True)

    y_train = data_train['# Letter'].values
    X_train = data_train.drop('# Letter', axis = 1)

    # Data normalization (0,1)
    X_train = preprocessing.normalize(X_train, norm='l2')

    # Models training

    # Neural Networks multi-layer perceptron (MLP) algorithm
    clf_NN = MLPClassifier(solver='adam',
                           activation='logistic',
                           alpha=0.001,
                           hidden_layer_sizes=(500,),
                           random_state=0,
                           max_iter=1000)

    clf_NN.fit(X_train, y_train)

    # Save model
    dump(clf_NN, bin_path+'\\NN_model.pkl')

if __name__ == '__main__':
    train()
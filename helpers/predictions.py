from collections.abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import os
import warnings

def predictionFunction(mode:str, trainingData:Iterable, classes:Iterable[int], classNames:Iterable[str], testData=None, testSize:float=0.2, layers:tuple=(8, 4), iterations:int=3200, outPath:str='data/predictions/', termPath:str='termPath/', randState:int=64):
    '''
        Holds all the prediction models inside. This is the main method
        with which the performance of a selected model is tested.

        mode: Required to select which model is used. Depending on
              selection, some function parameters might not be used.
              Options are: 'cnn', 'svc', 'dtc', 'cnb'

        trainingData: Numpy ndarray which is acquired by using
                      separateSequences()

        classes: List of classes corresponding to each entry in the
                 trainingData

        classNames: List of class names to represent each of the
                    classes. Amount of names in the list should be
                    equal to the amount of classes you have

        testData: Optional. Should be a Numpy ndarray which contains 
                  the validation data

        testSize: Ratio between training and validation data. Defaults
                  to 0.2

        layers: Tuple of hidden layers for use with the MLPClassifier.
                Defaults to (8,4)

        iterations: Number of iterations, only for use with LinearSVC
                    and MLPClassifier

        outPath: Root path for predictions. Defaults to "data/predictions"

        termPath: Path for predictions for a certain term. Defaults to
                  "termPath/", but should be changed unless this method
                  is being used to test a single model.

        randState: Integer to allow for reproducible predictions.
                   Defaults to 64
    '''
    if mode is not None:
        if not os.path.isdir(outPath):
            os.makedirs(outPath)
        
        if termPath == "termPath/":
            warnings.warn("termPath should not be left at default. This will overwrite any prediction, unless it's being used to test a single model.")
        
        x_train, x_test, y_train, y_test = train_test_split(trainingData, classes, test_size=testSize)

        scaler = StandardScaler(with_mean=False)
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(testData) if testData is not None else scaler.transform(x_test)

        if mode == 'cnn':

            print(f'Metrics for CNN prediction using {layers} for hidden layer sizes, {iterations} iterations')

            clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=layers , max_iter=iterations, random_state=randState)
            with open(outPath+termPath+mode+".txt", "w") as f:
                clf.fit(x_train, y_train)
                prediction = clf.predict(x_test)
                f.write(str(classification_report(y_test, prediction, zero_division=0.0, target_names=classNames)))

        elif mode == 'svc':
            print('Metrics for LinearSVC')
            svc = LinearSVC(max_iter=iterations*10, random_state=randState)
            with open(outPath+termPath+mode+".txt", "w") as f:
                svc.fit(x_train, y_train)
                prediction = svc.predict(x_test)
                f.write(str(classification_report(y_test, prediction, zero_division=0.0, target_names=classNames)))

        elif mode == 'dtc':
            print('Metrics for DecisionTreeClassifier')
            dtc = DecisionTreeClassifier(criterion='entropy', random_state=randState)
            with open(outPath+termPath+mode+".txt", "w") as f:
                dtc.fit(x_train, y_train)
                prediction = dtc.predict(x_test)
                f.write(str(classification_report(y_test, prediction, zero_division=0.0, target_names=classNames)))

        elif mode == 'cnb':
            print('Metrtics for CategoricalNB')
            cnb = CategoricalNB(min_categories=244)
            with open(outPath+termPath+mode+".txt", "w") as f:
                cnb.fit(x_train, y_train)
                prediction = cnb.predict(x_test)
                f.write(str(classification_report(y_test, prediction, zero_division=0.0, target_names=classNames)))

        else:
            raise Exception("Unsupported mode. Expected: \'cnn\', \'svc\', \'dtc\', or \'cnb\'. Got:", mode)
    else:
        raise Exception("No mode passed. Expected: \'cnn\', \'svc\', \'dtc\', or \'cnb\'.")

#--------------------------------------------------------------------------------------------------------------------------------------------------
    
def vectorizeData(kmerList, ngramRange:tuple=(4,4)):
    '''
        Returns a NumPy ndarray with vectorized k-mer sequences using the 
        TfidfVectorizer.

        This vectorizer was chosen due to its ability to normalize the frequencies
        of each k-mer, as is analog to real life frequencies in organisms. Some
        sequences and codon groups are rarer than others, so to be able to remove
        some of the biases towards one or another means that we can test the models
        in an orderly fashion without having any skewed data.
        
        kmerList: A complete list of kmers generated from the full list of sequences
                  obtained by using createData()
        
        ngramRange was added for finer control and testing for the vectorizer.
    '''
    vectorizer = TfidfVectorizer(ngram_range=ngramRange)
    return vectorizer.fit_transform(kmerList)
 

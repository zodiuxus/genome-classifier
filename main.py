from helpers import predictions as pred
from helpers import sequenceFetch as sf

if __name__ == '__main__':
    
    # window = 3
    # hostClassPairs = [("beak and feather disease virus", 4), ("influenza a virus", 7), ("agapornis roseicollis", 3), ("cacatua moluccensis", 5), ("avian paramyxovirus", 6)]
    # virusClassPairs = [("influenza a virus", 0), ("avian paramyxovirus", 1), ("beak and feather disease virus", 2)]
    # terms = ["agapornis roseicollis[Orgn]", "cacatua moluccensis[Orgn]", "beak and feather disease virus host", "influenza a virus host", "avian paramyxovirus complete genome", "beak and feather disease virus[Orgn]", "influenza a virus[Orgn]"]
    # If you're going to create a training and validation dataset that have similar names or entries,
    # I recommend you split the term-class pairs into multiple lists and outputting to different folders.
    # sf.getData(terms, 100, 10, "your email here")
    # sf.getSequences(hostClassPairs, outPath="data/sequences/hosts/")
    # sf.getSequences(virusClassPairs, outPath="data/sequences/viruses/")
    # sf.combineSequences(inPath="data/sequences/hosts/", outPath="data/partial_data/", outFile="hosts.txt", skipFirst=True)
    # sf.combineSequences(inPath="data/sequences/viruses/", outPath="data/partial_data/", outFile="viruses.txt", skipFirst=True)
    # sf.combineSequences(inPath="data/partial_data/")
    # sf.createKmers(windowSize=window)

    sequences, classes, classCounts = sf.separateSeqAndClass("kmers.txt")
    viralClassCount = classCounts[0]+classCounts[1]+classCounts[2]
    vectorizedData = pred.vectorizeData(sequences, (1, 4), 'cvec')
    viralData = vectorizedData[:viralClassCount, :]
    viralClasses = classes[:viralClassCount]
    hostData = vectorizedData[viralClassCount:, :]
    hostClasses = classes[viralClassCount:]
    '''
        Below are the 3 splitting methods in order.

        First is the naive one, with no previous separation between training and testing data

        Second is the most accurate to real-life classification of patients one, which
        uses the viral DNA as the training data, and each of the hosts separately as the
        validation data

        Third is a variation of the second, where the training data are the hosts, and the
        validation data are each of the viral genomes
    '''
    classNames = ["Alphainfluenzavirus", "Avian paramyxovirus", "Beak and feather disease virus", "Agapornis roseicollis", "BFDV Host", "Cacatua moluccensis", "Avian paramyxovirus Host", "IAV Host"]
    pred.predictionFunction('cnb', trainingData=vectorizedData.toarray(), classes=classes, testSize=0.2, classNames=classNames)
    pred.predictionFunction('cnn', trainingData=vectorizedData, classes=classes, testSize=0.2, classNames=classNames)
    pred.predictionFunction('dtc', trainingData=vectorizedData, classes=classes, testSize=0.2, classNames=classNames)
    pred.predictionFunction('svc', trainingData=vectorizedData, classes=classes, testSize=0.2, iterations=8192, classNames=classNames)

    offset = 0
    classNames = ["Alphainfluenzavirus", "Avian paramyxovirus", "Beak and feather disease virus"]
    for term, amount in [("Agapornis roseicollis", classCounts[3]), ("Cacatua moluccensis", classCounts[5]), ("BFDV Host", classCounts[4]), ("Influenza A virus Host", classCounts[7]), ("Avian paramyxovirus", classCounts[6])]:
        print("Predicting for " + term)
        curHost = viralData[offset:offset+amount, :]
        pred.predictionFunction('cnn', trainingData=viralData, classes=viralClasses, testSize=amount, classNames=classNames, termPath=term+"/", layers=(32, 16, 8), testData=curHost)
        pred.predictionFunction('dtc', trainingData=viralData, classes=viralClasses, testSize=amount, classNames=classNames, termPath=term+"/", testData=curHost)
        pred.predictionFunction('svc', trainingData=viralData, classes=viralClasses, testSize=amount, iterations=8192, classNames=classNames, termPath=term+"/", testData=curHost)
        offset+=amount

    offset = 0
    classNames = ["Agapornis roseicollis", "BFDV Host", "Cacatua moluccensis", "Avian paramyxovirus Host", "IAV Host"]
    for term, amount in [("Influenza A virus", classCounts[0]), ("Avian paramyxovirus", classCounts[1]), ("BFDV", classCounts[2])]:
        print("Predicting for " + term)
        curHost = viralData[offset:offset+amount, :]
        pred.predictionFunction('cnn', trainingData=hostData, classes=hostClasses, testSize=amount, classNames=classNames, termPath=term+"/", layers=(32, 16, 8), testData=curHost)
        pred.predictionFunction('dtc', trainingData=hostData, classes=hostClasses, testSize=amount, classNames=classNames, termPath=term+"/", testData=curHost)
        pred.predictionFunction('svc', trainingData=hostData, classes=hostClasses, testSize=amount, iterations=8192, classNames=classNames, termPath=term+"/", testData=curHost)
        offset+=amount

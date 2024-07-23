from helpers import predictions as pred
from helpers import sequenceFetch as sf


if __name__ == '__main__':
    
    window = 3
    hostClassPairs = [("beak and feather disease virus", 4), ("agapornis roseicollis", 3), ("cacatua moluccensis", 5), ("avian paramyxovirus", 6)]
    virusClassPairs = [("alphainfluenzavirus", 0), ("avian paramyxovirus", 1), ("beak and feather disease virus", 2)]
    terms = ["agapornis roseicollis[Orgn]", "cacatua moluccensis[Orgn]", "beak and feather disease virus host", "alphainfluenzavirus host", "avian paramyxovirus complete genome", "beak and feather disease complete genome", "alphainfluenzavirus complete genome"]
    # If you're going to create a training and validation dataset that have similar names or entries,
    # I recommend you split the term-class pairs into multiple lists and outputting to different folders.
    sf.getData(terms, 100, 10, "email goes here")
    sf.getSequences(hostClassPairs, outPath="data/sequences/hosts/")
    sf.getSequences(virusClassPairs, outPath="data/sequences/viruses/")
    sf.combineSequences(inPath="data/sequences/hosts/", outFile="hosts.txt")
    sf.combineSequences(inPath="data/sequences/viruses/", outFile="viruses.txt")
    sf.combineSequences(inPath="data/combined_data/")
    sf.createKmers(windowSize=window)

    sequences, classes, classCounts = sf.separateSeqAndClass("combined_sequences.txt", window)
    viralClassCount = classCounts[0]+classCounts[1]+classCounts[2]
    vectorizedData = pred.vectorizeData(sequences)
    viralData = vectorizedData[:viralClassCount, :]
    viralClasses = classes[:viralClassCount]
    hostData = vectorizedData[viralClassCount:, :]
    offset = 0
    classNames = ["Alphainfluenzavirus", "Avian paramyxovirus", "Beak and feather disease virus"]
    for term, amount in [("Agapornis roseicollis", classCounts[3]), ("BFDV", classCounts[4]), ("Cacatua moluccensis", classCounts[5]), ("Avian paramyxovirus", classCounts[6])]:
        print("Predicting for " + term)
        curHost = hostData[offset:offset+amount, :]
        ratio = curHost.shape[0]/viralClassCount
        pred.predictionFunction('cnb', trainingData=viralData.toarray(), testData=curHost.toarray(), classes=viralClasses, testSize=ratio, classNames=classNames, termPath=term)
        pred.predictionFunction('cnn', trainingData=viralData, testData=curHost, classes=viralClasses, testSize=ratio, classNames=classNames, termPath=term, layers=(32, 16, 8))
        pred.predictionFunction('dtc', trainingData=viralData, testData=curHost, classes=viralClasses, testSize=ratio, classNames=classNames, termPath=term)
        pred.predictionFunction('svc', trainingData=viralData, testData=curHost, classes=viralClasses, testSize=ratio, classNames=classNames, termPath=term)
        offset+=amount


import pandas as pd
import os
from Bio import SeqIO, Entrez 
from collections.abc import Iterable
import warnings
'''
    The order in which everything is fetched is:
        1. getData() - acquires entries based on the list of terms

        2. getSequences(): turns the sequence files into workable sequences,
                            mode can be changed to output to file for larger
                            sequence files

        3. combineSequences(): optional, should be used to combine all sequence
                               files if multiple are obtained/created

        4. createKmers(): creates k-mers based on parameter upon function call,
                          for a longer list of sequences it's recommended to
                          use the file output method, as it saves progress
                          and makes sure you don't run out of memory at any step

        5. separateSeqAndClass(): separates the k-mers and classes they belong
                                  to, while also returning the amount of entries
                                  each class has which is vital to the function
                                  of this application
'''

def getData(terms:Iterable, maxRecords:int, batchSize:int, email:str="A.N.Other@example.com", outPath:str="data/entries/", returnType:str="fasta"):
    '''
        This will export each entry's search results to a separate file.
        It does not support stitching together the files into one.

        terms: List of terms to search for

        maxRecords: Maximum amount of records to obtain per term

        batchSize: Amount of records to download per cycle

        email: Required to announce your identity to NCBI. Use something
               other than the default.

        outPath: The directory in which the downloaded files are put in.
                 The output file will be named after each term in the list.
                 Defaults to "data/entries/"
        
        returnType: Datatype to use. Defaults to "fasta"
        
    '''
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    if email == "A.N.Other@example.com":
        warnings.warn("Try not to use the default email, instead use your own. Always tell NCBI who you are in case of errors.")

    if not terms:
        raise Exception("List of terms should not be empty. Add some terms first, then run again.")

    record_ids = []
    Entrez.email = email
    for term in terms:
        print("Searching for %s" %term)
        stream = Entrez.esearch(db="nucleotide", term=term,retmax=str(maxRecords))
        record = Entrez.read(stream)
        record_ids = record['IdList']
        stream.close()

        count = len(record_ids)

        results = Entrez.read(Entrez.epost('nucleotide', id=','.join(record_ids)))
        webenv = results['WebEnv']
        query_key = results['QueryKey']

        with open(str(outPath) + str(term) +"."+returnType, "w") as output:
            for start in range(0, count, batchSize):
                end = min(count, start + batchSize)
                print("Downloading record %i to %i" % (start+1, end))
                stream = Entrez.efetch(db='nucleotide', rettype=returnType, retmode='text', retstart='start', retmax='batchSize', webenv=webenv, query_key=query_key)
                data = stream.read()
                if type(data) is str:
                    output.write(data)
                stream.close()

def createKmers(inPath:str="data/combined_data/", outPath:str="data/kmers/", inFile:str='combined_sequences.txt', outFile:str='kmers', sequences=[], windowSize:int=1, step:int=1, mode:str='l'):

    '''
        Generates a list of k-mers created from groups of k sequential nucleotides
        from each sequence.

        Output format is: outPath + windowSize_outFile ; outFile needs an extension.

        inPath: Input path from which sequences are gathered. Defaults to
        "data/combined_data/", which is the output of combineSequences()

        outPath: Path in which the k-mer file (if selected) are output to. Requires
                 outFile

        inFile: Input file from which sequences for processing are read. Defaults to
                "combined_sequences.txt" which is the output of combineSequences()

        outFile: File to ouput the generated k-mers within the outPath directory.
                  Defaults to "kmers", and is combined with windowSize

        windowSize: Amount of nucleotides in a row to be grouped to form a new sequence
        for training. Defaults to 1, which duplicates every nucleotide.

        step: Skip a certain amount of nucleotides on each sequence. Defaults to 1,
        which doesn't skip any nucleotide

        mode: Use 'l' to output to file, 's' to output to iterable. Defaults to 'l'


    '''
    
    if mode.lower() == 'l' and outFile!= '':
        with open(outPath+str(windowSize)+"_"+outFile, 'w') as output:
            output.write("sequence,class\n")
            
            data = pd.read_csv(inPath+inFile, sep=" ")
            data = data[data['sequence'].str.contains("seq") == False]
            sequences = data['sequence'].tolist()
            dataClasses = data['class'].tolist()
            for sequence in sequences:
                temp = ''
                for x in range(0, len(sequence)-windowSize+1, step):
                    temp += sequence[x:x+windowSize]+' '
                output.write(temp+','+str(dataClasses[sequences.index(sequence)])+"\n")

    elif mode.lower() == 's' and sequences:
        kmers = []
        for sequence in sequences:
            temp = ''
            for x in range(0, len(sequence)-windowSize+1, step):
                temp += sequence[x:x+windowSize]+' '
            kmers.append(temp)
        return kmers

def sequenceToFile(sequences, termClassPairs, outPath:str, outfile:str):
    '''
        Moves all the sequences acquired from getSequences() to a file. This method
        shouldn't be called separately.

        sequences: List of sequences obtained from getSequences(), though you won't
                   need to call it yourself. This is done directly from the
                   getSequences() function, and only if it's selected.

        outFile: Output file without extension for the finished sequences and their
                 classes. Retrieve these using pandas.read_csv() using " " as the
                 delimiter.
    '''
    seqs = []
    for item in sequences:
        unavailable = False
        for term, classNo in termClassPairs:
            if term.lower() in item.description.lower():
                seqs.append(item.seq+" "+str(classNo))
            elif not unavailable:
                unavailable = True
                print("The term and class pair for " + term + " - " + str(classNo) + "has not been found.")

    with open(outPath+outfile+".txt", "w") as output:
        print("Writing into "+outPath+outfile+".txt")
        for x in range(0, len(seqs)):
            output.write(str(seqs[x]).lower()+"\n")

def getSequences(termClassPairs:Iterable, mode:str='l', inPath:str="data/entries/", outPath:str="data/sequences/", datatype:str="fasta", fileName=None):
    
    '''
        This will take the relative path and file type on which to perform sequence
        extraction. By default, it will look for entries in the data/entries/
        directory, as is the standard output of the getData() method.
        
        termClassPairs: Used to give a term its corresponding class, as each term
                        is searched within the description of the acquired list of
                        entries. It is recommended to give each sequence group a
                        class, otherwise it may not be able to return an accurate
                        dataset

        fileName: The name of the input and output file post-transformation. By
                  default, it will extract the sequences from each file in the
                  "path" directory. Can accept a single file in string format or
                  a list or tuple of file names
        
        mode: Whether to output to file (l) or to output to list (s). File output
              will also require an output path, but one is provided by default

        inPath: Input directory, defaults to "data/entries/" which is the same as
                the output directory of getData()

        outPath: Output directory, defaults to "data/sequences/"

        datatype: Datatype matching the file type, defaults to "fasta"
    '''
    
    if fileName is str:
        sequences = list(SeqIO.parse(inPath+fileName, datatype))
        if mode.lower() == 'l':
            sequenceToFile(sequences, termClassPairs, outPath, fileName)

        elif mode.lower() == 's':
            return [str(x.seq).lower() for x in sequences]

    else:
        if fileName is None:
            files = os.listdir(inPath)
        else:
            files = fileName
        for file in files:
            sequences = list(SeqIO.parse(inPath+file, datatype))
            if mode.lower() == 'l':
                sequenceToFile(sequences, termClassPairs, outPath, file)

            elif mode.lower() == 's':
                return [str(x.seq).lower() for x in sequences]

def separateSeqAndClass(fileName:str, windowSize:int, inPath:str="data/kmers/"):
    '''
        Takes as input the output of createKmers as fileName and returns the
        k-mers and list of classes each k-mer belongs to, as well as the amount
        of entries for each class, sorted by number of class.

        windowSize: required to fetch the correct list of k-mers

        inPath: path in which the file is searched for, defaults to "data/kmers/"
    '''
    kmers = pd.read_csv(inPath+str(windowSize)+"_"+fileName, sep=',')
    kmers.sort_values(by='class', inplace=True)
    sequences = kmers['sequence'].tolist()
    classes = kmers['class'].tolist()
    return sequences, classes, kmers.value_counts('class', sort=False).tolist()

def combineSequences(inPath:str="data/sequences/", outPath:str="data/combined_data/", outFile:str="combined_sequences.txt"):
    '''
        Takes a directory of sequences and combines them into one large csv-like
        file.

        inPath: Directory of sequences yet to be combined, defaults to
                "data/sequences/"

        outPath: Directory of output file post combining, defaults to
                 "data/combined_data/"

        outFile: Name of file in the output directory, defaults to
                 "combined_sequences.txt"
    '''
    if not os.path.isdir(inPath):
        os.mkdir(inPath)
    fileNames = os.listdir(inPath)
    with open(outPath+outFile, "w") as outfile:
        outfile.write("sequence class")
        for file in fileNames:
            with open(inPath+file) as infile:
                for line in infile:
                    outfile.write(line)


import pandas as pd
import numpy as np
import sys
import re

def read_the_file(filepath):
    '''
    This function will read the file and retreive the basic dataset
    consisiting of the words and the corresponding sentiment in the
    following structure:
    [
        [sentiment,list of words]   # example1
        [sentiment,list of words]   # example2
    ]

    Also, it will give a list of the bag of words to make a proper
    word vector representation of each of the example.
    '''
    #Initializing the return structures
    dataset=[]
    word_bag={}

    #Creating the delimiters to split the string
    #we wont remove the ?,! since they could convey certain emotion
    delimeter=",|-| |;|\.|\(|\)|\n|\"|:|'|/|&|`|[|]"

    #Reading the file line by line
    print("Opening the datafile in read mode")
    total_pos=0
    total_neg=0
    with open(filepath,'r') as fhandle:
        # i=0
        # print(fhandle)
        for line in fhandle:
            # i+=1
            # print(line)
            # if(i==10):
            #     break
            #Splitting the words by the delimiters
            tokens=re.split(delimeter,line)
            #Now retreiving the class type and doc name
            filename=tokens[2]
            sentiment=tokens[1]
            #Removing the header of the review metadata
            tokens.pop(0)   #Removing the field of the review
            tokens.pop(0)   #Removing the sentiment type
            tokens.pop(0)   #Removing the file name
            tokens.pop(0)   #Removing the file extension
            #Testing the integrity of the filename and labels
            if(filename=='' or sentiment==''):
                print("Getting empty string in filename and sentiment")
                sys.exit(0)

            #Removing the empty token from the list and hashing the words
            new_tokens=[]
            for token in tokens:
                #Discarding the empty tokens
                if(token!='' and len(token)!=1):
                    new_tokens.append(token)
                    #Hashing or incrementing the word cound to bag
                    try:
                        word_bag[token]+=1
                    except:
                        word_bag[token]=1

            #Appening them to the example structure
            dataset.append([int(sentiment=="neg"),new_tokens])
            if(sentiment=="neg"):
                total_neg+=1
            else:
                total_pos+=1
            # print(dataset)
            # print(tokens)
            # print(word_bag)
            # sys.exit(0)

        #Finally printing the word bag to see the frequency of each word
        # #Seeing the frequency of the maximum frequcy ones
        # import operator
        # sorted_bag=sorted(word_bag.items(),key=operator.itemgetter(1))
        # total_word=0
        # for key,value in sorted_bag:
        #     print("word: {}\t count: {}".format(key,value))
        #     total_word+=1

        #Printing the stattistics of the dataset
        #print(dataset)
        print("\nTotal word Count:",len(word_bag.keys()))
        print("Total num of pos:{} and neg:{}".format(total_pos,\
                                                        total_neg))

        return total_pos,total_neg,dataset,word_bag


def get_example_and_label():
    '''
    This function will chop the line read from file and give the
    appropriate label and the training text (to be encoded later)
    '''


if __name__=="__main__":
    #file related attributes
    filepath="dataset/naive_bayes_data.txt"

    #Calling the transformation function
    read_the_file(filepath)

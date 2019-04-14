import pandas as pd
import numpy as np
import sys
import re

def get_example_and_label(filepath,split_factor):
    '''
    This function will read the file and retreive the basic dataset
    consisiting of the words and the corresponding sentiment in the
    following structure:

    dateset:
    [
        [sentiment,list of words]   # example1
        [sentiment,list of words]   # example2
        ...
        ...
    ]

    Also, it will give a list of the bag of words to make a proper
    word vector representation of each of the example.

    word_bag:
    {
        word_name1   : [pos_count,neg_count]
        word_name2   : [pos_count,neg_count]
        ...
        ...
    }
    The word bag will only add the enteries for the data in training
    set to estimate the word probability in both positive and negetive
    sentiment world.

    We will skip if any new words appear in the test dataset at test
    time. Actually this should be the approach to be followed in
    real world also.
    '''
    #Initializing the return structures
    train_dataset=[]
    test_dataset=[]
    train_word_bag={}

    #Creating the delimiters to split the string
    #we wont remove the ?,! since they could convey certain emotion
    delimeter=",|-| |;|\.|\(|\)|\n|\"|:|'|/|&|`|[|]|\{|\}"

    #Reading the file line by line
    print("Opening the datafile in read mode")
    total_pos=0
    total_neg=0
    ex_count=0
    with open(filepath,'r') as fhandle:
        for line in fhandle:
            #Creating a done flag hashing the elements no to count twice
            done_flag={}

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
                    if(ex_count%split_factor!=0):
                        #Checking if the word occurance is already reflected
                        try:
                            #If already counted then leave
                            _=done_flag[token]
                        except:
                            #Otherwise set the done flag and ranp up count
                            done_flag[token]=True
                            _fill_word_bag(token,sentiment,train_word_bag)

            #Appening them to the example structure
            if(ex_count%split_factor==0):
                test_dataset.append([int(sentiment=="neg"),new_tokens])
            else:
                train_dataset.append([int(sentiment=="neg"),new_tokens])
                if(sentiment=="neg"):
                    total_neg+=1
                else:
                    total_pos+=1

            #Finally incrementing the example count
            ex_count+=1

        #Finally printing the word bag to see the frequency of each word
        # #Seeing the frequency of the maximum frequcy ones
        import operator
        sorted_bag=sorted(train_word_bag.items(),key=operator.itemgetter(1))
        total_word=0
        for key,value in sorted_bag:
            if(value[0]+value[1]<9):
                print("word: {}\t count: {}".format(key,value))
                total_word+=1
        print("Total less frequent words:",total_word)

        return total_pos,total_neg,train_dataset,train_word_bag,\
                test_dataset

def _fill_word_bag(token,sentiment,word_bag):
    '''
    Helper function to fill the word to the word bag and incrementing the
    count appropriately either in the positive or the negetive class.
    '''
    field=int(sentiment=="neg")
    try:
        #incrementing the count of the word for its corresponding class
        word_bag[token][field]+=1
    except:
        #Initializing the word entries with zero
        word_bag[token]=[0,0]
        word_bag[token][field]+=1

def get_train_test_split(filepath,split_factor=5):
    '''
    This function will be the main handler of the dataset creation and
    splitting, and finally returning the train and validation split.

    split_factor: represent that we add to test_dataset every split
                    factor example number.
    '''
    #Reading the file and getting the train and test split
    dataset=get_example_and_label(filepath,split_factor)

    #Retreiving the dataset elements
    print("\nDataset Summary:")
    print("Total Positive Exmaple(training):",dataset[0])
    print("Total Negetive Example(training):",dataset[1])
    print("Total number of train example:",len(dataset[2]))
    print("Total number of test examples:",len(dataset[4]))

    return dataset


if __name__=="__main__":
    #file related attributes
    filepath="dataset/naive_bayes_data.txt"

    #Calling the transformation function
    get_train_test_split(filepath,split_factor=5)

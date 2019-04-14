import numpy as np
import pandas as pd
from data_handling import get_train_test_split

#Class Definition for the Naive Bayes Classifier
class NaiveBayes():
    #Training Attributes: for probability Estimation
    train_word_bag=None         #the word table with doc-count in
    train_dataset=None          #datast containing the label and review
    total_pos=None              #Total number of positive examples
    total_neg=None              #Total number of negetive examples

    #Trained Attributes: after prob estimation
    word_prob_world=None        #Prob of word occuring in given world
    pos_world_prob=None         #probability of seeing a pisitive review
    neg_world_prob=None         #Prob of the seeing a negetive review

    #Validation/Testing Attributes
    test_dataset=None

    #Initialization function
    def __init__(self,dataset):
        '''
        DESCRIPTION:
            The constructor for this class.
        '''
        print("\n\n#############################################")
        print("Customizing the classifier for your dataset")
        print("Assigning the training attributes to class members")
        #initializing the training datset
        self.total_pos=dataset[0]
        self.total_neg=dataset[1]
        self.train_dataset=dataset[2]
        self.train_word_bag=dataset[3]

        #Retreiving the validation datset
        self.test_dataset=dataset[4]

        #Now learning the class estimation for future is pssive way
        print("\nEstimating the word's probability in each world")
        self.estimate_word_prob()
        print("\nEstimating the probability of each world")
        self.estimate_world_prob()

    #Function to learning the probability of each word in respective world
    def estimate_word_prob(self):
        '''
        1. TRAIN time:
        This function will estimate the probability of occurance of
        a word in the world of a particular sentiment using the maximum
        likeliehood estimate as we did for the bernoulli trial.

        2. TEST time:
        The estimate will be done only for the data in training set and
        will not contain the words which might not be in training set.
        So, we will skip those words from the review assuming they are
        only specific to that particular class hence having probability
        in that world as 1.
        '''
        #Creating the probability dictionary
        word_prob_world={}
        #Calcualting the probability estimate
        for word,counts in self.train_word_bag.items():
            #Getting the estimate of occurance of word in each world
            #prob of occurance in world of positive sentiment
            prob_pos_world=float(counts[0]+1)/float(self.total_pos)
            prob_neg_world=float(counts[1]+1)/float(self.total_neg)

            #Finally saving the estimation in both world
            word_prob_world[word]=[prob_pos_world,\
                                    prob_neg_world]

            #Printing the probability
            # print("word:{0[0]}\tpos_prob:{0[1]:.6f}\tneg_prob:{0[2]:.6f}\t".format(\
            #                         [word,\
            #                         float(prob_pos_world),\
            #                         float(prob_neg_world)]))

        #Assing the classifier this estimation
        self.word_prob_world=word_prob_world

    #Function to estimate the probabiltiy of each review
    def estimate_world_prob(self):
        '''
        This function will estimate the probability of seeing reviews
        from each of the world itelf.
        This is highly dependent on the dataset, currently we have ensured
        that both the world have around equal number of reviews, so
        both the probability will be same.
        '''
        total_review = self.total_pos+self.total_neg
        #Calcualting the prob of each world
        self.pos_world_prob = self.total_pos/total_review
        self.neg_world_prob = self.total_neg/total_review

        print("Probability of pos_world: ",self.pos_world_prob)
        print("Probability of neg world: ",self.neg_world_prob)

    #Function to evaluate the accurcy of dataset
    def evaluate_accuracy(self,mode):
        '''
        Given a dataset this function will evalaute the performance of the
        naive bayes learned on the training data.
        '''
        #Intializing the elements of the confuction matrix
        #For positive labels
        true_pos=0
        false_neg=0
        #For negetive labels
        false_pos=0
        true_neg=0

        #Initializing the dataset based on the mode
        print("\n#############################################")
        dataset=None
        if(mode=="train"):
            print("Evaluating the train dataset")
            dataset=self.train_dataset
        else:
            print("Evalauting the test dataset")
            dataset=self.test_dataset

        #Now iterating over the examples
        for example in dataset:
            #Retreiving the labels and review from the example
            label=example[0]
            review=example[1]
            # print("LABEL: ",label)
            #Evaluating the review
            eval=self.evaluate_review(review)
            # print("PREDICTED: {}\n".format(eval))

            if(label==0 and eval==0):
                true_pos+=1;
            elif(label==0 and eval==1):
                false_neg+=1
            elif(label==1 and eval==0):
                false_pos+=1
            elif(label==1 and eval==1):
                true_neg+=1
        #Printing the confusion matrix and accuracy
        self.print_eval_metric(true_pos,false_neg,false_pos,true_neg)
        print("#############################################")

    #Function to evaluate the world of review from learned params
    def evaluate_review(self,review):
        '''
        Given a particular review this function will evaluate the whether
        it belongs to world1 or world2 (positive or negetive reviews).
        '''
        #Intializing the prob of this review
        rev_pos_prob=1.0*self.pos_world_prob
        rev_neg_prob=1.0*self.neg_world_prob

        #Iteraiting over the review one by one
        for word in review:
            #Retreiving the word prob if it exist in our dictionary
            try:
                word_pos_prob = self.word_prob_world[word][0]
                word_neg_prob = self.word_prob_world[word][1]
            except:
                #Skipping this word otherwise
                continue

            #Appending the contribution of this word to rev prob
            #print(word,word_pos_prob,word_neg_prob)
            rev_pos_prob = rev_pos_prob*word_pos_prob
            rev_neg_prob = rev_neg_prob*word_neg_prob

        #Finally we have the unnormalized review probability
        norm = rev_pos_prob+rev_neg_prob+1e-40
        #print(norm)
        if(rev_pos_prob>rev_neg_prob):
            # print("pos_rev: un_prob:{} \t prob:{}".format(\
                                    # rev_pos_prob,rev_pos_prob/norm))
            return 0
        else:
            # print("neg_rev: un_prob:{} \t prob:{}".format(\
                                    # rev_neg_prob,rev_neg_prob/norm))
            return 1

    #Function to print the evaluation metric
    def print_eval_metric(self,true_pos,false_neg,false_pos,true_neg):
        '''
        This function will print the following metric:

        1. Confusion metric:
                    prediction-0    prediction-1
        label-0    |     TP       |      FN     |
        label-1    |     FP       |      TN     |

        2. Accuracy:
                    (TP+TN)/(TP+FN+FP+TN)
        3. Precision:
                    (TP)/(TP+FP)
        4. Recall:
                    (TP)/(TP+FN)
        '''
        print("\nConfuction Matrix:")
        print("\tpred-0\tpred-1")
        print("label-0\t{}\t{}".format(true_pos,false_neg))
        print("label-1\t{}\t{}\n".format(false_pos,true_neg))

        print("Accuracy: {}\n".format(float(true_pos+true_neg)/\
                            float(true_pos+false_neg+false_pos+true_neg)))

        print("Precision: {}\n".format(float(true_pos)/\
                            float(true_pos+false_pos)))

        print("Recall: {}\n".format(float(true_pos)/\
                            float(true_pos+false_neg)))


if __name__=="__main__":
    #Getting the dataset
    filepath="dataset/naive_bayes_data.txt"
    dataset=get_train_test_split(filepath,split_factor=5)

    #Now creating the NaiveBayes classifier object
    myBayes=NaiveBayes(dataset)
    #Testing on training dataset
    myBayes.evaluate_accuracy(mode="train")
    #Testing on testing dataset
    myBayes.evaluate_accuracy(mode="test")

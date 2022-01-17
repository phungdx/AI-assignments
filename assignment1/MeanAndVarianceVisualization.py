'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Chris Clingerman
'''
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape
    

    trials = 2
    k_fold = 5
    accuracies_pure = []
    accuracies_stump = []
    accuracies_dt3 = []
    accuracies_dt2 = []
    accuracies_dt5 = []
    iterate=0
    for i in range(trials):
        idx = np.arange(n)
        np.random.seed(i)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        fold_size = int(n / k_fold)
        for j in range(k_fold):

            if j == (k_fold - 1):
                
                Xtest = X[j*fold_size:, :]
                Xtrain = X[0: j*fold_size, :]
                ytest = y[j*fold_size:, :]
                ytrain = y[0: j*fold_size, :]

            else:
                Xtest = X[j*fold_size: j*fold_size + fold_size, :]
                Xtrain = np.concatenate((X[0: (j-1)*fold_size + fold_size, :],X[j*fold_size + fold_size:, :]),axis=0)
                ytest = y[j*fold_size: j*fold_size + fold_size, :]
                ytrain = np.concatenate((y[0: (j-1)*fold_size + fold_size, :],y[j*fold_size + fold_size:, :]),axis=0)

            percent = 0.1
            for _ in range(10):
                Xtrain_sample = Xtrain[0:int(percent*n),:]
                ytrain_sample = ytrain[0:int(percent*n),:]
                percent += 0.1

                # train the decision trees
                clf_pure = tree.DecisionTreeClassifier()
                clf_stump = tree.DecisionTreeClassifier(max_depth=1)
                clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)
                clf_dt2 = tree.DecisionTreeClassifier(max_depth=2)
                clf_dt5 = tree.DecisionTreeClassifier(max_depth=5)

                clf_pure = clf_pure.fit(Xtrain_sample,ytrain_sample)
                clf_stump = clf_stump.fit(Xtrain_sample,ytrain_sample)
                clf_dt3 = clf_dt3.fit(Xtrain_sample,ytrain_sample)
                clf_dt2 = clf_dt2.fit(Xtrain_sample,ytrain_sample)
                clf_dt5 = clf_dt5.fit(Xtrain_sample,ytrain_sample)

                # output predictions on the remaining data
                y_pred_pure = clf_pure.predict(Xtest)
                y_pred_stump = clf_stump.predict(Xtest)
                y_pred_dt3 = clf_dt3.predict(Xtest)
                y_pred_dt2 = clf_dt2.predict(Xtest)
                y_pred_dt5 = clf_dt5.predict(Xtest)


                # compute the training accuracy of the models
                DecisionTreeAccuracy = accuracy_score(ytest, y_pred_pure)
                DecisionStumpAccuracy = accuracy_score(ytest, y_pred_stump)
                DT3Accuracy = accuracy_score(ytest, y_pred_dt3)
                DT2Accuracy = accuracy_score(ytest, y_pred_dt2)
                DT5Accuracy = accuracy_score(ytest, y_pred_dt5)

                accuracies_pure.append(DecisionTreeAccuracy)
                accuracies_stump.append(DecisionStumpAccuracy)
                accuracies_dt3.append(DT3Accuracy)
                accuracies_dt2.append(DT2Accuracy)
                accuracies_dt5.append(DT5Accuracy)


                # compute the training accuracy of the model
                meanDecisionTreeAccuracy = np.mean(accuracies_pure)
                stddevDecisionTreeAccuracy = np.std(accuracies_pure)

                meanDecisionStumpAccuracy = np.mean(accuracies_stump)
                stddevDecisionStumpAccuracy = np.std(accuracies_stump)

                meanDT3Accuracy = np.mean(accuracies_dt3)
                stddevDT3Accuracy = np.std(accuracies_dt3)

                meanDT2Accuracy = np.mean(accuracies_dt2)
                stddevDT2Accuracy = np.std(accuracies_dt2)

                meanDT5Accuracy = np.mean(accuracies_dt5)
                stddevDT5Accuracy = np.std(accuracies_dt5)


                plt.subplot(3,3,1)
                plt.errorbar(iterate,meanDecisionTreeAccuracy, yerr=stddevDecisionTreeAccuracy,fmt='.', color="r",elinewidth=1)
                plt.xlabel('Iteration')
                plt.ylabel('mean average accuracy')
                plt.ylim([0.5,1])
                plt.title('pure tree')

                plt.subplot(3,3,2)
                plt.errorbar(iterate,meanDecisionStumpAccuracy, yerr=stddevDecisionStumpAccuracy,fmt='.', color="g",elinewidth=1)
                plt.xlabel('Iteration')
                plt.ylabel('mean average accuracy')
                plt.ylim([0.5,1])
                plt.title('stump tree')

                plt.subplot(3,3,3)
                plt.errorbar(iterate,meanDT3Accuracy, yerr=stddevDT3Accuracy,fmt='.', color="b",elinewidth=1)
                plt.xlabel('Iteration')
                plt.ylabel('mean average accuracy')
                plt.ylim([0.5,1])
                plt.title('3-depth tree')

                plt.subplot(3,3,4)
                plt.errorbar(iterate,meanDT2Accuracy, yerr=stddevDT2Accuracy,fmt='.', color="y",elinewidth=1)
                plt.xlabel('Iteration')
                plt.ylabel('mean average accuracy')
                plt.ylim([0.5,1])
                plt.title('2-depth tree',y=-0.5)

                plt.subplot(3,3,5)
                plt.errorbar(iterate,meanDT5Accuracy, yerr=stddevDT5Accuracy,fmt='.', color="c",elinewidth=1)
                plt.xlabel('Iteration')
                plt.ylabel('mean average accuracy')
                plt.ylim([0.5,1])
                plt.title('5-depth tree',y=-0.5)

                iterate+=1


    # plt.legend()
    plt.show()






    # shuffle the data
    # idx = np.arange(n)
    # np.random.seed(13)
    # np.random.shuffle(idx)
    # X = X[idx]
    # y = y[idx]
    
    # # split the data
    # Xtrain = X[1:101,:]  # train on first 100 instances
    # Xtest = X[101:,:]
    # ytrain = y[1:101,:]  # test on remaining instances
    # ytest = y[101:,:]

    # # train the decision tree
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(Xtrain,ytrain)

    # # output predictions on the remaining data
    # y_pred = clf.predict(Xtest)




    # # compute the training accuracy of the model
    # meanDecisionTreeAccuracy = sum(accuracies_pure) / len(accuracies_pure)

    
    # # TODO: update these statistics based on the results of your experiment
    # stddevDecisionTreeAccuracy = np.std(accuracies_pure)
    # meanDecisionStumpAccuracy = sum(accuracies_stump) / len(accuracies_stump)
    # stddevDecisionStumpAccuracy = np.std(accuracies_stump)
    # meanDT3Accuracy = sum(accuracies_dt3) / len(accuracies_dt3)
    # stddevDT3Accuracy = np.std(accuracies_dt3)


    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    # evaluatePerformance()
    stats = evaluatePerformance()
    print ("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
    print ("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
    print ("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.
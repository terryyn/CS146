"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        self.probabilities_ = dict()
        count = Counter(y)
        total = sum(count.values())
        for key in count:
            self.probabilities_[key] = count[key]/total
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        y = np.random.choice(list(self.probabilities_.keys()),X.shape[0],p=list(self.probabilities_.values()))        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size,random_state = i)
        clf.fit(X_train,y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train_error += 1 - metrics.accuracy_score(y_train, train_pred, normalize=True)
        test_error += 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)
    train_error /= float(ntrials)
    test_error /= float(ntrials)
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    # #========================================
    # # part a: plot histograms of each feature
    # print('Plotting...')
    # for i in range(d) :
    #     plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf= RandomClassifier()
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf= DecisionTreeClassifier(criterion="entropy")
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    '''
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    '''


    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error knn3: %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error knn5: %.3f' % train_error)

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X,y)
    y_pred = clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error knn7: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    clf = MajorityVoteClassifier()
    train_error,test_error = error(clf,X,y)
    print('\t majority -- training error: %.3f  test error: %.3f' %(train_error,test_error))

    clf = RandomClassifier()
    train_error,test_error = error(clf,X,y)
    print('\t random -- training error: %.3f  test error: %.3f' %(train_error,test_error))
    
    clf = DecisionTreeClassifier(criterion="entropy")
    train_error,test_error = error(clf,X,y)
    print('\t decision tree -- training error: %.3f  test error: %.3f' %(train_error,test_error))

    clf = KNeighborsClassifier(n_neighbors=5)
    train_error,test_error = error(clf,X,y)
    print('\t KNN -- training error: %.3f  test error: %.3f' %(train_error,test_error))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    knns = []  
    errors = []
    for k in range(1,50,2):
        knns.append(k)
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn,X,y,cv=10)
        errors.append(1 - sum(score)/float(len(score)))
    best_error = min(errors)
    print('best error rate is '+str(best_error))
    print('best k for knn is ')
    for index in range(len(errors)):
        if errors[index] == best_error:
            print(knns[index])
    plt.plot(knns,errors)
    plt.xlabel("number of neighbors")
    plt.ylabel("error rate")
    plt.legend()
    plt.show()
    

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    depths = []
    train_errors = []
    test_errors = []
    for i in range(1,21):
        depths.append(i)
        clf_dt = DecisionTreeClassifier(criterion="entropy",max_depth=i)
        train_error,test_error = error(clf_dt,X,y)
        train_errors.append(train_error)
        test_errors.append(test_error)
    plt.plot(depths,train_errors,color="blue",label="train errors")
    plt.plot(depths,test_errors,color="red",label="test errors")
    plt.xlabel("depths")
    plt.ylabel("error rate")
    plt.legend()
    plt.show()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.9,random_state = 1000)
    splits = []
    knn_train = []
    knn_test = []
    dt_train = []
    dt_test = []
    for i in range(1,11):
        i = i/float(10)
        splits.append(i)
        knn_train_err = knn_test_err = dt_train_err = dt_test_err = 0
        for j in range(10):
            if i!= 1.0:
                X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_train,y_train,train_size = i,random_state = j)
            else:
                X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_train,y_train,train_size = 0.99,random_state = j)
            temp1,temp2= error(KNeighborsClassifier(n_neighbors=7),X_train_i,y_train_i)
            knn_train_err += temp1
            knn_test_err += temp2
            temp3,temp4 = error(DecisionTreeClassifier(criterion="entropy",max_depth=3),X_train_i,y_train_i)
            dt_train_err += temp3
            dt_test_err += temp4
        knn_train.append(knn_train_err/100)
        knn_test.append(knn_test_err/100)
        dt_train.append(dt_train_err/100)
        dt_test.append(dt_test_err/100)
    plt.plot(splits,knn_train,color="red",label="knn train")
    plt.plot(splits,knn_test,color="blue",label = "knn_test")
    plt.plot(splits,dt_train,color='yellow',label="dt_train")
    plt.plot(splits,dt_test,color='green',label="dt_test")
    plt.legend()
    plt.xlabel('portion of training data')
    plt.ylabel('error rate')
    plt.show()

    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()

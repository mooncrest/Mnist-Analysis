import data

import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import label_binarize

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k, method='decrease'):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        minDistIndex = np.argpartition(self.l2_distance(test_point), k)[:k]
        counts = np.bincount(np.take(self.train_labels, minDistIndex).astype('int64'))
        possible_digits = np.where(counts == np.amax(counts))[0]

        if len(possible_digits) == 1:
            return possible_digits[0]

        if method == 'decrease':
            return self.query_knn(test_point, k - 1)

        elif method == 'random':
            return np.random.choice(possible_digits) 

    def query_knn_proba(self, test_point, k):
        minDistIndex = np.argpartition(self.l2_distance(test_point), k)[:k]
        counts = np.bincount(np.take(self.train_labels, minDistIndex).astype('int64'), minlength=10) / k
        return counts

    def plot_ROC(self, test_data, test_labels, k):
        probs = []
        for data in test_data:
            probs.append(self.query_knn_proba(data, k))
        prediction_probs = np.array(probs)
        onehot_labels = label_binarize(test_labels, classes=[i for i in range(10)])

        for num in range(10):
            col_probs = prediction_probs[:, num]
            col_labels = onehot_labels[:,num]
            fpr, tpr, _ = roc_curve(col_labels, col_probs)
            plt.plot(fpr, tpr, marker='.', label=f"ROC curve for num {num} (AUC={auc(fpr, tpr)}")

        plt.xlim([-0.01, 0.4])
        plt.ylim([0.65, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()


    def plot_Confusion_Matrix(self, test_data, test_labels, k):
        prediction = np.array([self.query_knn(point, k) for point in test_data])
        CM = confusion_matrix(test_labels, prediction, labels= [i for i in range(10)])
        plt.imshow(CM, cmap=plt.cm.Blues)
        for x in range(10):
            for y in range(10):
                plt.text(y, x, f"{CM[x][y]}", horizontalalignment="center", color="green")

        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.xlabel("prediction")
        plt.ylabel("Actual Value")
        plt.show()

    def get_recall(self, test_data, test_labels, k):
        prediction = np.array([self.query_knn(point, k) for point in test_data])
        CM = confusion_matrix(test_labels, prediction, labels= [i for i in range(10)])
        denom = np.array([i if i != 0 else 1 for i in np.sum(CM, axis=1)])
        return np.diag(CM) / denom


    def get_precision(self, test_data, test_labels, k):
        prediction = np.array([self.query_knn(point, k) for point in test_data])
        CM = confusion_matrix(test_labels, prediction, labels= [i for i in range(10)])
        denom = np.array([i if i != 0 else 1 for i in np.sum(CM, axis=0)])
        return np.diag(CM) / denom
        
def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=10)
    accuracies = []
    for k in k_range:
        total_acc = 0
        for train, test in kf.split(train_data):

            test_labels = train_labels[test]
            test_data = train_data[test]

            knn_labels = train_labels[train]
            knn_data = train_data[train]

            knn = KNearestNeighbor(knn_data, knn_labels)

            total_acc += classification_accuracy(knn, k, test_data, test_labels)
            
        accuracies.append((k, total_acc / 10))

    return accuracies

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    result = []
    for data in eval_data:
        result.append(knn.query_knn(data, k))

    return np.sum(np.array(result) == eval_labels) / len(eval_labels)

def main():
    K = 3
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    #accuracy
    accuracy = classification_accuracy(knn, K, test_data, test_labels)
    print(f"achieved {accuracy} accuracy on test set with k = {K}")

    # ROC
    knn.plot_ROC(test_data, test_labels, K)

    # confusion matrix
    knn.plot_Confusion_Matrix(test_data, test_labels, K)

    # recall and precision
    recalls = knn.get_recall(test_data, test_labels, K)
    precisions = knn.get_precision(test_data, test_labels, K)
    for num in range(10):
        print(f"{num} has a recall of {recalls[num]} and precision {precisions[num]}")

if __name__ == '__main__':
    main()

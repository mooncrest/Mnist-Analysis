'''
IF tensor flow is not installed please comment out the import
and also in the if __name__ == '__main__' set run_NN to false
'''


import data

import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import label_binarize

import itertools

class SVM(object):
    def __init__(self, short=False):
        self.parameters = [
            {
                'kernel': ('poly', 'rbf'),
                'C': (1, 5),
                'probability': (True,)
            },
            {
                'kernel': ('poly',),
                'C': (1, ),
                'degree': (1, 5),
                'gamma': (0.0001, ),
                'probability': (True,)
            }
        ]
        if short:
            self.parameters = [{'kernel': ('linear',), 'C': [1],'probability': (True,)}]

        self.model = None

    def train_model(self, train_data, train_labels):
        """trains the svm model on the hyperparameters defined in initalizer"""
        svc = svm.SVC()
        # running whole parameter set takes too long only using set 1 and 2
        clf = GridSearchCV(svc, self.parameters, n_jobs=6)
        clf.fit(train_data, train_labels)

        print(f"best params for SVM: {clf.best_params_}")
        
        means = clf.cv_results_['mean_test_score']
        for mean, params in zip(means, clf.cv_results_['params']):
            print(f"accuracy of {mean} for {params}")

        self.model = clf
        return clf

    def test_model(self, test_data, test_labels):
        """tests the svm model"""
        predictions = self.model.predict_proba(test_data)
        accuracy = np.sum(np.equal(np.argmax(predictions, axis=1), test_labels)) / len(test_labels)
        return accuracy

    def plot_ROC(self, test_data, test_labels):
        prediction_probs = self.model.predict_proba(test_data)
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


    def plot_Confusion_Matrix(self, test_data, test_labels):
        predictions = np.argmax(self.model.predict_proba(test_data), axis=1)
        CM = confusion_matrix(test_labels, predictions, labels= [i for i in range(10)])
        plt.imshow(CM, cmap=plt.cm.Blues)
        for x in range(10):
            for y in range(10):
                plt.text(y, x, f"{CM[x][y]}", horizontalalignment="center", color="green")

        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.xlabel("prediction")
        plt.ylabel("Actual Value")
        plt.show()

    def get_recall(self, test_data, test_labels):
        predictions = np.argmax(self.model.predict_proba(test_data), axis=1)
        CM = confusion_matrix(test_labels, predictions, labels= [i for i in range(10)])
        denom = np.array([i if i != 0 else 1 for i in np.sum(CM, axis=1)])
        return np.diag(CM) / denom


    def get_precision(self, test_data, test_labels):
        predictions = np.argmax(self.model.predict_proba(test_data), axis=1)
        CM = confusion_matrix(test_labels, predictions, labels= [i for i in range(10)])
        denom = np.array([i if i != 0 else 1 for i in np.sum(CM, axis=0)])
        return np.diag(CM) / denom


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    svm = SVM(True)
    svm.train_model(train_data, train_labels)

    # accuracy
    accuracy = svm.test_model(test_data, test_labels)
    print(f"achieved {accuracy} accuracy on test set", end=" ")
    print(f"on parameters {svm.model.best_params_}")

    # ROC
    svm.plot_ROC(test_data, test_labels)

    # confusion matrix
    svm.plot_Confusion_Matrix(test_data, test_labels)

    # recall and precision
    recalls = svm.get_recall(test_data, test_labels)
    precisions = svm.get_precision(test_data, test_labels)
    for num in range(10):
        print(f"{num} has a recall of {recalls[num]} and precision {precisions[num]}")


if __name__ == '__main__':
    main()

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

class AdaBoost(object):
    def __init__(self, short=False):


        self.parameters = [
            {
                'n_estimators': [250, 300],
                'learning_rate': [1, 0.75, 0.5, 0.25],
                'algorithm': ["SAMME.R"]
            }, 
            {
                'n_estimators': [150],
                'learning_rate': [1, 0.75, 0.5],
                'algorithm': ["SAMME.R"]
            },
        ]

        if short:
            self.parameters = [{'n_estimators': [50], 'learning_rate': [1], 'algorithm': ["SAMME.R"]}]

        self.models = None

    def train_submodels(self, train_data, train_labels, num):
        """trains a specific submodel for num with all hyperparameters defined in initalizer"""
        new_label = np.array([1 if label == num else 0 for label in train_labels])

        AdaBoost = AdaBoostClassifier()
        clf = GridSearchCV(AdaBoost, self.parameters, n_jobs=6)
        clf.fit(train_data, new_label)

        print(f"best params for classifying {num}: {clf.best_params_}", end=' ')
        print(f"with average accuracy of {clf.best_score_} in 5 folds")

        return clf


    def train_model(self, train_data, train_labels):
        """trains 10 submodels each to classify each number"""
        self.models = [self.train_submodels(train_data, train_labels, i) for i in range(10)]
        return self.models

    def test_model(self, test_data, test_labels):
        """tests this model"""
        probabilities = []
        for model in self.models:
            probabilities.append(model.predict_proba(test_data)[:,1])

        predictions = np.argmax(np.array(probabilities), axis=0)
        accuracy = np.sum(np.equal(predictions, test_labels)) / len(predictions)
        return accuracy

    def plot_ROC(self, test_data, test_labels):
        probabilities = []
        for model in self.models:
            probabilities.append(model.predict_proba(test_data)[:,1])
        prediction_probs = np.array(probabilities).transpose()
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
        probabilities = []
        for model in self.models:
            probabilities.append(model.predict_proba(test_data)[:,1])

        predictions = np.argmax(np.array(probabilities), axis=0)

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
        probabilities = []
        for model in self.models:
            probabilities.append(model.predict_proba(test_data)[:,1])

        predictions = np.argmax(np.array(probabilities), axis=0)

        CM = confusion_matrix(test_labels, predictions, labels= [i for i in range(10)])
        denom = np.array([i if i != 0 else 1 for i in np.sum(CM, axis=1)])
        return np.diag(CM) / denom


    def get_precision(self, test_data, test_labels):
        probabilities = []
        for model in self.models:
            probabilities.append(model.predict_proba(test_data)[:,1])

        predictions = np.argmax(np.array(probabilities), axis=0)

        CM = confusion_matrix(test_labels, predictions, labels= [i for i in range(10)])
        denom = np.array([i if i != 0 else 1 for i in np.sum(CM, axis=0)])
        return np.diag(CM) / denom

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    adaBoost = AdaBoost(True)
    adaBoost.train_model(train_data, train_labels)

    # accuracy
    accuracy = adaBoost.test_model(test_data, test_labels)
    print(f"achieved {accuracy} accuracy on test set")

    # ROC
    adaBoost.plot_ROC(test_data, test_labels)

    # confusion matrix
    adaBoost.plot_Confusion_Matrix(test_data, test_labels)

    # recall and precision
    recalls = adaBoost.get_recall(test_data, test_labels)
    precisions = adaBoost.get_precision(test_data, test_labels)
    for num in range(10):
        print(f"{num} has a recall of {recalls[num]} and precision {precisions[num]}")

if __name__ == "__main__":
    main()
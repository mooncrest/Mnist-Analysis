import data

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import label_binarize

import itertools

class NeuralNet(object):
    def __init__(self, short=False):
        self.model = None
        self.params = None

        self.learning_rate = np.logspace(-3,1, 5)
        self.epochs = [i for i in range(3,7)]
        self.optimizer = [
            tf.keras.optimizers.Adam,
        ]
        self.loss = [
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ]

        if short:
            self.learning_rate = [0.001]
            self.epochs = [5]

    def permute_hyperparameters(self):
        return list(itertools.product(
                self.learning_rate,
                self.epochs,
                self.optimizer,
                self.loss
            )
        )

    def create_model(self):
        return tf.keras.models.Sequential([
            tf.keras.Input(shape=(64,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def train_model(self, train_data, train_labels):
        """trains the neural network"""

        best_acc = -1
        for lr, epoch, optimizer, loss in self.permute_hyperparameters():
            model = self.create_model()
            model.compile(
                optimizer=optimizer(learning_rate=lr),
                loss=loss,
                metrics=['accuracy']
            )
            hist = model.fit(train_data, train_labels, epochs=epoch, verbose=0)
            if hist.history['accuracy'][-1] > best_acc:
                best_acc = hist.history['accuracy'][-1]
                self.model = model
                self.params = {'learning_rate': lr, 'epochs': epoch}

        return model

    def test_model(self, test_data, test_labels):
        """tests the best neural network trained"""
        return self.model.evaluate(test_data, test_labels, verbose=0)

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

    NN = NeuralNet(True)
    NN.train_model(train_data, train_labels)

    # accuracy
    accuracy = NN.test_model(test_data, test_labels)
    print(f"achieved {accuracy[1]} accuracy on test set")
    print(f"with params {NN.params}")

    # ROC
    NN.plot_ROC(test_data, test_labels)

    # confusion matrix
    NN.plot_Confusion_Matrix(test_data, test_labels)

    # recall and precision
    recalls = NN.get_recall(test_data, test_labels)
    precisions = NN.get_precision(test_data, test_labels)
    for num in range(10):
        print(f"{num} has a recall of {recalls[num]} and precision {precisions[num]}")

if __name__ == '__main__':
    main()

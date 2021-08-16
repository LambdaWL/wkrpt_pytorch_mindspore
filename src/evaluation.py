import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class Metrics:
    """ Collecting accuracy measures. """
    def __init__(self, padding_label=-1):
        self._losses = []
        self._labels = []
        self._predictions = []

        self.padding_label = padding_label
    
    def update(self, loss, labels, predictions):
        """
        Args:
            loss: scalar
            labels: array of shape (B, qlen)
            predictions: array of shape (B, qlen)
        """
        self._losses.append(loss)
        
        # Remove padded labels 
        for label, prediction in zip(labels, predictions):
            for l, p in zip(label, prediction):
                if l != self.padding_label:
                    self._labels.append(l)
                    self._predictions.append(p)
    
    def evaluate(self):
        losses = np.array(self._losses)
        labels = np.array(self._labels)
        predictions = np.array(self._predictions)

        avg_loss = round(np.sum(losses) / len(losses), 4)
        avg_accuracy = round(accuracy_score(labels, predictions) * 100, 2)
        avg_fscore = f1_score(labels, predictions, average="weighted")
        avg_fscore = round(avg_fscore * 100, 2)

        self._losses = []
        self._labels = []
        self._predictions = []
        return avg_loss, avg_accuracy, avg_fscore

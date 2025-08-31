import numpy as np
from prettytable import PrettyTable
from matplotlib import pyplot as plt

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # Initialize confusion matrix with all zeros
        self.num_classes = num_classes  # Number of classes (5 classes in this dataset)
        self.labels = labels  # Class labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred is prediction result, labels are true labels
            self.matrix[p, t] += 1  # Count based on prediction and true label values, increment corresponding position in confusion matrix

    def summary(self):  # Calculate metrics function
        # calculate accuracy
        sum_TP = 0
        # Calculate total number of test samples
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # Sum of diagonal elements in confusion matrix, which is the number of correctly classified samples
        acc = sum_TP / n  # Overall accuracy
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # Create a table
        table.field_names = ["", "Accuracy", "Precision", "Recall", "F1-score"]
        TPT, FPT, FNT, TNT, F1_score = 0, 0, 0, 0, 0
        for i in range(self.num_classes):  # Calculate precision, recall, specificity
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            TPT += TP
            FPT += FP
            FNT += FN
            TNT += TN
        Precision = round(TPT / (TPT + FPT), 6) if TPT + FPT != 0 else 0.
        Recall = round(TPT / (TPT + FNT), 6) if TPT + FNT != 0 else 0.  # Accuracy for each class
        Specificity = round(TNT / (TNT + FPT), 6) if TNT + FPT != 0 else 0.
        F1_score = round(2*(Precision*Recall)/(Precision+Recall), 6)

        table.add_row(["value", acc, Precision, Recall, F1_score])
        print(table)
        # return str(acc)
        return acc, Precision, Recall, F1_score

    def plot(self):  # Plot confusion matrix
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # Set x-axis coordinate labels
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # Set y-axis coordinate labels
        plt.yticks(range(self.num_classes), self.labels)
        # Display colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # Annotate count/probability information in the plot
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # Note: matrix[y, x] here, not matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
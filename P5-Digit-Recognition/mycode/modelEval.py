from keras.models import Model
import operator

def accuracy(clf, dataset, sequences, labels):
    pred = clf.predict(dataset)
    acc = 0
    for sample in range(len(sequences)):
        pred_length = pred[0][sample]
        index0, value0 = max(enumerate(pred_length), key=operator.itemgetter(1))
        pred_d1 = pred[1][sample]
        index1, value1 = max(enumerate(pred_d1), key=operator.itemgetter(1))
        pred_d2 = pred[2][sample]
        index2, value2 = max(enumerate(pred_d2), key=operator.itemgetter(1))
        pred_d3 = pred[3][sample]
        index3, value3 = max(enumerate(pred_d3), key=operator.itemgetter(1))
        pred_d4 = pred[4][sample]
        index4, value4 = max(enumerate(pred_d4), key=operator.itemgetter(1))
        pred_d5 = pred[5][sample]
        index5, value5 = max(enumerate(pred_d5), key=operator.itemgetter(1))

        if (sequences[sample, index0] == 1 and labels[sample, index1, 0] == 1 and 
            labels[sample, index2, 1] == 1 and labels[sample, index3, 2] == 1 and 
            labels[sample, index4, 3] == 1 and labels[sample, index5, 4] == 1):
            acc += 1
    return 1. * acc / len(sequences)

import numpy as np

def top_5(pred, real):
    counter = 0.0
    counterr = 0.0
    counter_ten= 0.0
    for i in range(pred.shape[0]):
        if np.argmax(pred[i,:])==np.argmax(real[i,:]):
            counter+=1
        sort = np.flip(np.argsort(pred[i,:]))
        holder = np.isin(np.argmax(real[i,:]),sort[:5])
        holder_ten = np.isin(np.argmax(real[i,:]),sort[:10])
        if holder:
            counterr+=1
        if holder_ten:
            counter_ten+=1
    accuracy = counter/pred.shape[0]
    accuracy_five = counterr/pred.shape[0]
    accuracy_ten = counter_ten/pred.shape[0]
    return accuracy, accuracy_five, accuracy_ten

def evaluation(vectors_real,vectors_new):

    count = 0
    total = 0
    for i in range(vectors_real.shape[0]):
        for j in range(vectors_real.shape[0]):
            if j>i:
                errivsi = np.corrcoef(vectors_new[i,:],vectors_real[i,:])
                errivsj = np.corrcoef(vectors_new[i,:],vectors_real[j,:])
                errjvsi = np.corrcoef(vectors_new[j,:],vectors_real[i,:])
                errjvsj = np.corrcoef(vectors_new[j,:],vectors_real[j,:])

                if (errivsi[0,1] + errjvsj[0,1]) > (errivsj[0,1] + errjvsi[0,1]):
                    count+=1
                total+=1

    accuracy = count/total
    return accuracy
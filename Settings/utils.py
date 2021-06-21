import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def get_roc_curve(labels, predicted_vals, generator,name, roc_5 = True):
    auc_roc_vals = []
    y = np.concatenate([y for x, y in generator], axis=0)
    if roc_5:
        x = [2, 5, 6, 8, 10]
    else:
        x = range(len(labels))
    for k, i in enumerate(x):
        try:
            gt = y[:, i]

            pred = predicted_vals[:, i]

            auc_roc = roc_auc_score(gt, pred)

            auc_roc_vals.append(auc_roc)

            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[k] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(f'Error')
            #print(f"Error: {labels[i]}. ")
    plt.savefig(name)
    plt.show()
    return auc_roc_vals




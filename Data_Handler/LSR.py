import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


CLASSES =  ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
           'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
           'Fracture', 'Support Devices']

def uniform_vs_normal_distribution_showcase():
    uniform_dist = np.random.uniform(0.55, 0.85, 100)
    sns.kdeplot(uniform_dist, x="Values")
    plt.show()

    mu = (0.55+0.85)/2
    sigma = mu - 0.55
    #print(f'mu:{mu}, sigma = {sigma}')
    normal_dist = np.random.normal(mu, sigma, 100)
    sns.kdeplot(normal_dist, x="Values")
    plt.show()

def generate_uniform_dist_labels(a, b, number_of_labels = 139958):
    uniform_dist = np.random.uniform(a, b, number_of_labels) # 0.55, 0.85 # 0.0 0.3 # 0.05 0.3
    sns.kdeplot(uniform_dist, x="Values")
    plt.savefig("uniform")
    return uniform_dist

def get_number_of_labels(data_frame):
    df = pd.read_csv(data_frame)

    U_zeros = df.copy()
    U_ones = df.copy()

    for i in CLASSES:
        U_zeros[i][U_zeros[i] < 0] = 0.0
    U_zeros.to_csv("Train_U_zeros_LSR.csv", index=None)

    for i in CLASSES:
        U_ones[i][U_ones[i] < 0] = 1.0
    U_ones.to_csv("Train_U_ones_LSR.csv", index=None)
    print(U_zeros.head())
    print(U_ones.head())

def create_LSR(data_frame):
    df = pd.read_csv(data_frame)
    indexes = [12403, 8087, 5598, 1488, 12984, 27742, 18770, 33739, 3145, 11628, 2653, 642, 1079]
    distribution = generate_uniform_dist_labels(0.55, 0.85, number_of_labels = 139958)
    k = 0
    U_zeros = df.copy()

    for idx, i in enumerate(CLASSES):
        upper = indexes[idx] +k
        U_zeros[i][U_zeros[i] < 0] = distribution[k:upper]
        k = upper

    U_zeros.to_csv("Train_U_ones_LSR_005_03.csv", index=None)


create_LSR('patients_data_updated.csv')




'''df = pd.read_csv('Train_U_zeros_LSR_0_03.csv')
U_zeros = df.copy()
number_of_uncrt =[]
for i in CLASSES:
    # print(U_zeros[i])
    # print("\n")
    # print(U_zeros[i][U_zeros[i] < 0])
    # print('\n')

    quant = U_zeros[i][U_zeros[i].between(0.00000000000000001,0.35)]
    print(len(quant))
    number_of_uncrt.append(len(quant))
    print('\n')
print(number_of_uncrt)
with open("file2.txt", "w") as output:
    output.write(str(number_of_uncrt))
    output.write(str('\n'))
    output.write(str(sum(number_of_uncrt)))'''

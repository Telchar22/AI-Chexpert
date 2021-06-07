import pandas as pd
from Settings import settings as s
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def path_update(path):
    '''
    Updates paths to images.
    :return:
    '''
    input_df = pd.read_csv(path)
    input_df.fillna(0, inplace=True)
    input_df.drop(['Sex', 'Age', 'AP/PA'], axis=1, inplace=True)

    def vectorize_path_expand(x):
        return s.abs_path + x

    input_df['Path'] = vectorize_path_expand(input_df['Path'])
    input_df.to_csv(s.patients_workfile, index=None)

class DataSplit(object):
    def __init__(self, df, X_train, X_test, IDdf = None):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.IDdf = IDdf

    def add_ID_to_each_patient(self):
        y = self.df
        ID = []
        k = 0
        tmp = ''
        for i in y['Path']:
            p = Path(i).parts
            if tmp != p[4]:
                k += 1
            ID.append(k)
            tmp = p[4]
        y['ID'] = ID
        #y.to_csv('data_test.csv', index=False)
        print(y.head())
        self.IDdf = y

    def patient_overlap(self):
        # check if any patient appear in more like 1 data set.
        def path(x):
            p = Path(x).parts
            return p[4]

        df_t = self.X_train.copy()
        df_te = self.X_test.copy()
        df_t['Path'] = df_t['Path'].apply(path)
        df_te['Path'] = df_te['Path'].apply(path)
        # create set of extracted patient id's for the training set
        ids_train_set = set(df_t['Path'].values)
        # create set of extracted patient id's for the test set
        ids_test_set = set(df_te['Path'].values)
        # extract list of ID's that overlap in 2 sets
        patient_overlap = list(ids_train_set.intersection(ids_test_set))
        i = patient_overlap
        print(f'There are {len(i)} overlaping patients.')
        if len(i) != 0:
            print('')
            print(f'These patients are in 2 datasets:')
            print(f'{i}')

    def data_split(self):
        df2 = self.df

        X = df2
        gs = GroupShuffleSplit(n_splits=2, train_size=.98, random_state=33)
        train_ix, test_ix = next(gs.split(X, groups=X.ID))

        self.X_train = X.loc[train_ix]
        self.X_test = X.loc[test_ix]

        # X_train.to_csv('Tr.csv', index=False)
        # X_test.to_csv('Ts.csv', index=False)


class DataInfo(object):
    '''
      Data plotting and checking
    '''

    def __init__(self, df,name,  dict2=None):
        self.df = df
        self.dict2 = dict2
        self.name = name

    def prepare_dicts(self):
        print("Prepare data\n")
        # self.df=df
        print(self.df.head())
        dict2 = {}
        for i in s.CLASSES:
            dict2[i] = {0.0: 0, 1.0: 0, -1.0: 0}

        tmp_data = {}
        for i in s.CLASSES:
            tmp_data[i] = self.df[i].value_counts().to_dict()
            dict2[i].update(tmp_data[i])
        self.dict2 = dict2

    def data_checking(self):
        '''
        Some data checking after formatting.
        :return:
        '''
        new_dict2 = self.dict2
        print(new_dict2)
        for key, value in new_dict2.items():
            print("\nDisease ID:", key)
            if value[0.0] == new_dict2[key][0.0]:
                print("correct 0.0")
            else:
                print("fail 0.0")
            if value[1.0] == new_dict2[key][1.0]:
                print("correct 1.0")
            else:
                print("fail 1.0")
            if -1.0 in new_dict2[key]:
                if value[-1.0] == new_dict2[key][-1.0]:
                    print("correct -1.0")
                else:
                    print("fail -1.0")
            else:
                print('not -1.0')

    def plot(self):
        new_dict2 = self.dict2
        x = np.arange(len(s.CLASSES))  # the label locations
        width = 0.4  # the width of the bars

        y2 = [new_dict2[x][1.0] for x in s.CLASSES]
        y3 = [new_dict2[x][-1.0] for x in s.CLASSES]

        fig, ax = plt.subplots()
        fig.set_size_inches(19, 15)

        rects2 = ax.bar(x - width / 2, y2, width, label='positive')
        rects3 = ax.bar(x + width / 2, y3, width, label='uncertain')

        ax.set_ylabel('Quantity', fontsize=32)
        ax.set_title('Disease label distribution', fontsize=32)
        ax.set_xticks(x)
        ax.set_xticklabels(s.CLASSES, rotation=90) #, fontsize=18
        ax.legend(fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.bar_label(rects2, padding=3, fontsize=18)
        ax.bar_label(rects3, padding=3, fontsize=18)
        ax.set_yscale('log')
        #plt.savefig(self.name+'_pos_unc.png',bbox_inches='tight')
        plt.show()

    def plot_pos_neg(self):
        new_dict2 = self.dict2
        x = np.arange(len(s.CLASSES))  # the label locations
        width = 0.4  # the width of the bars

        y2 = [new_dict2[x][1.0] for x in s.CLASSES]
        y3 = [new_dict2[x][0.0] for x in s.CLASSES]

        fig, ax = plt.subplots()
        fig.set_size_inches(19, 15)

        rects2 = ax.bar(x - width / 2, y2, width, label='positive')
        rects3 = ax.bar(x + width / 2, y3, width, label='negative')

        ax.set_ylabel('Quantity', fontsize=32)
        ax.set_title('Disease label distribution', fontsize=32)
        ax.set_xticks(x)
        ax.set_xticklabels(s.CLASSES, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(fontsize=24)

        ax.bar_label(rects2, padding=3, fontsize=18)
        ax.bar_label(rects3, padding=3, fontsize=18)
        ax.set_yscale('log')
        #plt.savefig(self.name+'_pos_neg.png',bbox_inches='tight')
        plt.show()

    def stats(self):
        new_dict2 = self.dict2
        tmp = []
        tmp2 = []
        for key, value in new_dict2.items():
            print(key)
            for k, v in value.items():
                if k != 0.0:
                    print(f'{k}: {round((v / 223414) * 100, 2)}%')
                if k == 1.0:
                    tmp.append(round((v / 223414), 4))
                if k == -1.0:
                    tmp2.append(round((v / 223414), 4))
        print(tmp)
        print(tmp2)
        data = dict(zip(s.CLASSES, list(map(lambda num: round((num * 100), 2), tmp))))
        print(dict(sorted(data.items(), key=lambda item: item[1])))

        N = len(s.CLASSES)
        positive = list(map(lambda num: round((num * 100), 2), tmp))
        uncertain = list(map(lambda num: num * 100, tmp2))

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35
        fig = plt.figure()
        fig.set_size_inches(16, 14)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(ind, positive, width, color='r')
        ax.bar(ind, uncertain, width, bottom=positive, color='b')
        ax.set_ylabel('Percentage of labels relative to the total set [%]', fontsize=24)
        ax.set_title('Data distribution', fontsize=32)
        ax.set_xticks(ind)
        ax.set_xticklabels(s.CLASSES, fontsize=24, rotation=90)
        ax.set_yticks(np.arange(0, 55, 2.5))
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.legend(labels=['positive', 'uncertain'], fontsize=24)

        #plt.savefig(self.name + '_data_distribution.png',bbox_inches='tight')
        plt.show()




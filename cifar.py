import pickle
import numpy as np
import os

class Cifar100:
    def __init__(self):
        with open('data/glove.npy', 'rb',) as f:
            self.train = np.load(f, encoding='latin1',allow_pickle=True)
        with open('data/label.npy', 'rb', ) as f:
            self.train_label = np.load(f, encoding='latin1', allow_pickle=True)
        self.train_data = self.train  #训练数据
        self.train_labels = self.train_label  #数据标签
        self.train_groups, self.val_groups, self.test_groups = self.initialize()
        self.batch_num = 5

    def initialize(self):
        train_groups = [[],[],[],[],[]]
        for train_data, train_label in zip(self.train_data, self.train_labels):
            train_data = train_data[:928].reshape(29, 32)
            if int(train_label) < 20:
                train_groups[0].append((train_data,train_label))
            elif 20 <= int(train_label) < 40:
                train_groups[1].append((train_data,train_label))
            elif 40 <= int(train_label) < 60:
                train_groups[2].append((train_data,train_label))
            elif 60 <= int(train_label) < 80:
                train_groups[3].append((train_data,train_label))
            elif 80 <= int(train_label) < 100:
                train_groups[4].append((train_data,train_label))
        assert len(train_groups[0]) == 12000, len(train_groups[0])
        assert len(train_groups[1]) == 12000, len(train_groups[1])
        assert len(train_groups[2]) == 12000, len(train_groups[2])
        assert len(train_groups[3]) == 12000, len(train_groups[3])
        assert len(train_groups[4]) == 12000, len(train_groups[4])

        val_groups = [[],[],[],[],[]]
        test_groups = [[], [], [], [], []]
        for i, train_group in enumerate(train_groups):
            val_groups[i] =train_groups[i][450:500]+train_groups[i][1050:1100]+train_groups[i][1650:1700]+train_groups[i][2250:2300]+ train_groups[i][2850:2900]+train_groups[i][3450:3500]+train_groups[i][4050:4100]+train_groups[i][4650:4700] + train_groups[i][5250:5300]+train_groups[i][5850:5900]+train_groups[i][6450:6500]+train_groups[i][7050:7100] + train_groups[i][7650:7700]+train_groups[i][8250:8300]+train_groups[i][8850:8900]+train_groups[i][9450:9500] + train_groups[i][10050:10100]+train_groups[i][10650:10700]+train_groups[i][11250:11300]+train_groups[i][11850:11900]
            test_groups[i] = train_groups[i][500:600] + train_groups[i][1100:1200] + train_groups[i][1700:1800]+train_groups[i][2300:2400]+ train_groups[i][2900:3000] + train_groups[i][3500:3600] + train_groups[i][4100:4200]+train_groups[i][4700:4800] +train_groups[i][5300:5400] + train_groups[i][5900:6000] + train_groups[i][6500:6600]+train_groups[i][7100:7200] + train_groups[i][7700:7800] + train_groups[i][8300:8400] + train_groups[i][8900:9000]+train_groups[i][9500:9600] + train_groups[i][10100:10200] + train_groups[i][10700:10800] + train_groups[i][11300:11400]+train_groups[i][11900:12000]
            train_groups[i] = train_groups[i][0:450] + train_groups[i][600:1050] + train_groups[i][1200:1650]+train_groups[i][1800:2250] + train_groups[i][2400:2850] + train_groups[i][3000:3450] + train_groups[i][3600:4050]+train_groups[i][4200:4650] + train_groups[i][4800:5250] + train_groups[i][5400:5850] + train_groups[i][6000:6450]+train_groups[i][6600:7050] + train_groups[i][7200:7650]+train_groups[i][7800:8250] + train_groups[i][8400:8850] + train_groups[i][9000:9450]+train_groups[i][9600:10050] + train_groups[i][10200:10650] + train_groups[i][10800:11250] + train_groups[i][11400:11850]
        assert len(train_groups[0]) == 9000
        assert len(train_groups[1]) == 9000
        assert len(train_groups[2]) == 9000
        assert len(train_groups[3]) == 9000
        assert len(train_groups[4]) == 9000
        print(len(train_groups[0]))
        print(len(val_groups[0]))
        assert len(val_groups[0]) == 1000
        assert len(val_groups[1]) == 1000
        assert len(val_groups[2]) == 1000
        assert len(val_groups[3]) == 1000
        assert len(val_groups[4]) == 1000

        print(len(test_groups[0]))
        assert len(test_groups[0]) == 2000   #50*4=200
        assert len(test_groups[1]) == 2000
        assert len(test_groups[2]) == 2000
        assert len(test_groups[3]) == 2000
        assert len(test_groups[4]) == 2000

        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

if __name__ == "__main__":
    cifar = Cifar100()
    print(len(cifar.val_groups[0]))

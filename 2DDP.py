import numpy as np
import math, os
import random 
import multiprocessing as mp

class TwoDPercolation():
    def __init__(self, LENGTH, select_min=0, select_max=None):
        self.LENGTH= LENGTH
        self.z = 1.58
        self.select_min = select_min
        if select_max == None:
            self.select_max = np.math.ceil(self.LENGTH**self.z)

    def TwoDdoPercolationStep(self, arr, PROP, time):
        ODD_STEP = 2
        even = time%ODD_STEP
        arr_copy = np.copy(arr)
        row = len(arr_copy)
        for i in range(even, self.LENGTH, ODD_STEP):
#            for j in range(even, self.LENGTH, ODD_STEP):
            if arr[i] == 1:
                pro_left = random.random()
                pro_right = random.random()

                if pro_left < PROP:
                    arr_copy[(i+row-1)%row] = 1
                if pro_right < PROP:
                    arr_copy[(i+1)%row] = 1
                arr_copy[i] = 0
        return arr_copy

    def DP(self, p):
        vector_LT = []
        arr = np.ones(shape=[self.LENGTH])
        vector_LT.append(arr)
        for i in range(0, self.select_max):
            arr = self.TwoDdoPercolationStep(arr, p, i)
            vector_LT.append(arr)
        return vector_LT[self.select_min:]

    def configuration(self, num_sample, p, num_thread):
        pool = mp.Pool(num_thread)
        P = [p for i in range(num_sample)]
        config = pool.map(self.DP, P)
        return config

if __name__ == '__main__':
    prob = 0.01
    p_train = [round(prob*i,3) for i in range(int(1/prob))]
    print(p_train)
    LENGTH = 64
    TwoDP = TwoDPercolation(LENGTH, select_min=0, select_max=None)
    num_sample = 1000
    num_thread = 30
    for p in p_train:
        if not os.path.exists('2D_percolation/'+str(LENGTH)+'/'+str(p)+'.npy'):
            try:
                os.makedirs('2D_percolation/'+str(LENGTH))
            except:
                pass
            config = TwoDP.configuration(num_sample, p, num_thread)
            np.save('2D_percolation/'+str(LENGTH)+'/'+str(round(p, 3)), config)

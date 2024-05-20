import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def plot_xy(q,i):
    plt.cla()
    plt.plot(q, i) 
    plt.xlabel('log(q(A^-1))')
    plt.ylabel('log(I(q))')
    plt.title('q(A^-1)  vs  I(q)')
    plt.show()


def read_data(dat_file):
    with open(dat_file, 'r') as file:
        text = file.read()
        text = (text.split('\n'))[:-1]
        all_list = []
        for line in text:
            if line[0] != '%':
                try:
                    line = [eval(i) for i in line.split()]
                    all_list.append(line)
                except:
                    pass
        all_list = np.array(all_list)
        
        q = all_list[20:3020,0].copy()
        i = all_list[20:3020,1].copy()
        return q, i 

if __name__ == "__main__":
    dat_file = r"/home/p51pro/UD/jayraman_lab/MURI_Additive/SAXS_data_test/test_data/S1CJT016_4degC_25.0C_00377_00001.dat"   
    q, i = read_data(dat_file)
    i = np.log10(i)
    q = np.log10(q)
    
    print(i.shape)
    plot_xy(q,i)
    


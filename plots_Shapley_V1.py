import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

def plot(RESULTPATH):
    dir = RESULTPATH + '/' + 'log'
    focusvar = ["variational", "deterministic"]
    focus_k = ["1"]
    focus_linear = ['linear', 'nonlinear']
    filenames = os.listdir(dir )
    #print(filenames)
    plt.figure()

    info = {}
    for i, filename in enumerate(filenames):
        data = pd.read_csv(dir + '/' + filename)
        parts = filename.split("=")
        print(parts)
        avg = parts[0].split("_")[0]
        trainepochs= parts[1].split("_")[0]
        var ="variational" if parts[2].split("_")[0] == "True" else "deterministic"
        lin = parts[3].split("_")[1].split(".csv")[0]
        k = parts[3].split("_")[0]
        #print(k, var, lin)
        #print(var)
        if var in focusvar and k in focus_k and lin in focus_linear and int(trainepochs)< 100:
            plt.plot(data.iloc[:30,1], label = avg+var+lin+k+"_" +trainepochs)
            plt.legend()

    plt.show()

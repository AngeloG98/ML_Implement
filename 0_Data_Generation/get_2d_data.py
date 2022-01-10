from typing import NewType
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_input_pos():
    N = 2000
    plt.figure('Input Points')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('add points = MouseLEFT \n pop points = MouseRIGHT \n  finish = MouseMIDDLE') 
    pos = plt.ginput(N, timeout=1000)
    plt.close()
    poslist = [list(pos[i]) for i in range(len(pos))]
    return np.array(poslist)

def get_input_neg(pos):
    N = 2000
    plt.figure('Input Points')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('add points = MouseLEFT \n pop points = MouseRIGHT \n  finish = MouseMIDDLE')
    plt.scatter(pos[:,0],pos[:,1])
    neg = plt.ginput(N, timeout=1000)
    plt.close()
    neglist = [list(neg[i]) for i in range(len(neg))]
    return np.array(neglist)

def save_linear_data():
    pos = get_input_pos()
    df = pd.DataFrame(columns = ["X","Y"], data = pos)
    df.to_csv("./0_Data_Generation/data/linear_data.csv",encoding='utf-8',index=False)

def save_nonlinear_data():
    pos = get_input_pos()
    df = pd.DataFrame(columns = ["X","Y"], data = pos)
    df.to_csv("./0_Data_Generation/data/nonlinear_data.csv",encoding='utf-8',index=False)

def save_linear_logistic_data():
    pos = get_input_pos()
    neg = get_input_neg(pos)
    pos_tag = np.column_stack((pos,np.ones(pos.shape[0])))
    neg_tag = np.column_stack((neg,np.zeros(neg.shape[0])))
    pos_neg = np.vstack((pos_tag,neg_tag))
    np.random.shuffle(pos_neg)
    df = pd.DataFrame(columns = ["X1","X2","Tag"],data = pos_neg)
    df.to_csv("./0_Data_Generation/data/linear_classification_data.csv",encoding='utf-8',index=False)

def save_nonlinear_logistic_data():
    pos = get_input_pos()
    neg = get_input_neg(pos)
    pos_tag = np.column_stack((pos,np.ones(pos.shape[0])))
    neg_tag = np.column_stack((neg,np.zeros(neg.shape[0])))
    pos_neg = np.vstack((pos_tag,neg_tag))
    np.random.shuffle(pos_neg)
    df = pd.DataFrame(columns = ["X1","X2","Tag"],data = pos_neg)
    df.to_csv("./0_Data_Generation/data/nonlinear_classification_data.csv",encoding='utf-8',index=False)

if __name__ == "__main__":
    # save_linear_data()
    # save_nonlinear_data()
    # save_linear_logistic_data()
    save_nonlinear_logistic_data()
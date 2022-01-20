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

def get_imput_multi(a, b):
    N = 2000
    plt.figure('Input Points')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title('add points = MouseLEFT \n pop points = MouseRIGHT \n  finish = MouseMIDDLE')
    plt.scatter(a[:,0],a[:,1])
    plt.scatter(b[:,0],b[:,1],c="green")
    c = plt.ginput(N, timeout=1000)
    plt.close()
    clist = [list(c[i]) for i in range(len(c))]
    return np.array(clist)

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
    pos_tag = np.column_stack((pos,np.ones(pos.shape[0]))) # 1
    neg_tag = np.column_stack((neg,np.zeros(neg.shape[0]))) # 0
    pos_neg = np.vstack((pos_tag,neg_tag))
    np.random.shuffle(pos_neg)
    df = pd.DataFrame(columns = ["X1","X2","Tag"],data = pos_neg)
    df.to_csv("./0_Data_Generation/data/linear_classification_data.csv",encoding='utf-8',index=False)


def save_classification_data(type = "sep"):
    pos = get_input_pos()
    neg = get_input_neg(pos)
    pos_tag = np.column_stack((pos,np.ones(pos.shape[0]))) # 1
    neg_tag = np.column_stack((neg,-1*np.ones(neg.shape[0]))) # -1
    pos_neg = np.vstack((pos_tag,neg_tag))
    np.random.shuffle(pos_neg)
    df = pd.DataFrame(columns = ["X1","X2","Tag"],data = pos_neg)
    if type == "sep": #separable linear
        filename = "./0_Data_Generation/data/linear_svm_data.csv"
    elif type == "nonsep": #non-separable linear
        filename = "./0_Data_Generation/data/nonsep-linear_svm_data.csv"
    elif type == "nonlinear": #nonlinear
        filename = "./0_Data_Generation/data/nonlinear_svm_data.csv"
    elif type == "dt": #decision tree
        filename = "./0_Data_Generation/data/decision_tree_data.csv"
    df.to_csv(filename,encoding='utf-8',index=False)

def save_multiclass_data(type):
    a = get_input_pos()
    b = get_input_neg(a)
    c = get_imput_multi(a, b)
    a_tag = np.column_stack((a,np.ones(a.shape[0]))) # 1
    b_tag = np.column_stack((b,-1*np.ones(b.shape[0]))) # -1
    c_tag = np.column_stack((c,np.zeros(c.shape[0]))) # 0
    abc = np.vstack((a_tag,b_tag,c_tag))
    np.random.shuffle(abc)
    df = pd.DataFrame(columns = ["X1","X2","Tag"],data = abc)
    if type == "dt":
        df.to_csv("./0_Data_Generation/data/multiclass_data.csv",encoding='utf-8',index=False)
    elif type == "knn":
        df.to_csv("./0_Data_Generation/data/knn_data.csv",encoding='utf-8',index=False)


if __name__ == "__main__":
    # save_linear_data()
    # save_nonlinear_data()
    # save_linear_logistic_data()
    # save_classification_data("dt")
    save_multiclass_data("knn")
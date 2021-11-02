""" 将hen_diag过滤出来的病人与hkb41910中的人字段结合 """
import numpy as np
import pandas as pd


if __name__ == '__main__':
    hen_diag_path = '../data/diag_selection/Coronary artery disease diag records.json'
    df = pd.read_json(hen_diag_path)

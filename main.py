import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns

train, test = pd.read_csv('./data/train.csv'), pd.read_csv('./data/test.csv')

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
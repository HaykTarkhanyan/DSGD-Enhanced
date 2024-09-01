I have removed 16 rows from breast-cancer-wisconsin.data due to missing data (some ?-n)

now there are 685 rows left

In wines dataset I replaced white with 1 and red with 0

Made all the label columns in data_binary be "labels"

df_  = pd.read_csv(r"datasets\breast-cancer-wisconsin.csv")
df_["labels"] = df_["labels"].replace({2: 0, 4: 1})


Datasets added, Sept 1, 2024
https://www.kaggle.com/datasets/uciml/german-credit
https://www.kaggle.com/datasets/uciml/adult-census-income
https://archive.ics.uci.edu/dataset/222/bank+marketing

Activate the `thesis` environment 


## Some docs on files:
utlis.py - Has most of the core functions for our approach, also some rule generation helper functions
run_clustering.ipynb - At first I run this to get 


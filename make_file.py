import pandas as pd

path_1 = r'C:\Users\Administrator\Desktop\Edo_dir\MSc\5 year\MLDM\project_MLDM\a_1.csv'
path_2 = r'C:\Users\Administrator\Desktop\Edo_dir\MSc\5 year\MLDM\project_MLDM\a_2.csv'
path_3 = r'C:\Users\Administrator\Desktop\Edo_dir\MSc\5 year\MLDM\project_MLDM\data.csv'
path_names = r'C:\Users\Administrator\Desktop\Edo_dir\MSc\5 year\MLDM\project_MLDM\adult.csv'
df1 = pd.read_csv(path_1,  header=None)
df2 = pd.read_csv(path_2,  header=None)
df_names = pd.read_csv(path_names)
names = df_names.columns.tolist()

dfs = [df1, df2]
a = pd.concat(dfs, axis=0)
print(names, type(names))
a.columns = names
print(a)
a.to_csv(path_3, index=False, index_label=False)
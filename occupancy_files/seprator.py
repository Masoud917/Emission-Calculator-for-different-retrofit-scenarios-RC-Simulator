import pandas as pd

df = pd.read_excel('source.xlsx', header=None)
df = df.drop(df.columns[0], axis=1)

# print(df.head())

def repeat_row_in_column(row_index, repeat_count):
    row_to_repeat = df.iloc[row_index]
    result= []
    for i in range(repeat_count):
        result.extend(row_to_repeat.tolist())
    return result

weeks= 0
columns_data= []
while weeks < 52:
    columns_data.extend(repeat_row_in_column(2, 5))
    columns_data.extend(repeat_row_in_column(3, 1))
    columns_data.extend(repeat_row_in_column(4, 1))
    weeks+= 1
columns_data.extend(repeat_row_in_column(2, 1))
# 8760 hours or one year
df= pd.DataFrame({"people":columns_data})
df.to_excel('library.xlsx', index=False, header=False)
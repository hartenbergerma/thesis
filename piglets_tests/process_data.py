import os
import pandas as pd
import scipy.io


# open xlsx as csv file using pandas
xlsx_path = "datasets/piglet/overview.xlsx"
df = pd.read_excel(xlsx_path, sheet_name="Sheet1", engine="openpyxl")

# iterate through all rows
for i in range(len(df)):
    filename = df.spectrum[i]
    path = f"datasets/piglet/{df.piglet[i]}"

    img_mat = scipy.io.loadmat(path + "/" + filename)

    # # check if DarkCount is part of the Ws file
    # if "DarkCount" in img_mat.keys():
    #     print(f"{filename} contains DarkCount")
    # else:
        # print(f"{filename} does not contain DarkCount")
    
    # # find DarkCount file with same date
    # date = filename.split("_")[2].split(".")[0].split(" ")[0].split("_")[0]
    # for file in os.listdir(path):
    #     if date in file and "DarkCount" in file:
    #         print(f"{file}")
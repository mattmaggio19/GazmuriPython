

import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def BleedingPlots():
    path = r"C:\Users\mattm\Documents\Gazmuri Docs\Bleeding at 15 min in HSTBI model.xlsx"
    df = pd.read_excel(path, skiprows = [0,1])
    print(df.columns.values)
    print(df["Seconds into experiment that we reached 940g blood removed"])
    plt.subplot(1,3,1)
    plt.title("Weight of blood removed at 15 min (g)")
    data = df["Weight of blood removed at 15 min"]
    plt.boxplot(df["Weight of blood removed at 15 min"][0:len(data)-1])
    plt.scatter(1, 940, color='r', s=24)

    plt.subplot(1,3,2)
    plt.title("Final weight of blood")
    data = df["Final weight of blood"]
    plt.boxplot(data[0:len(data)-1] )

    plt.scatter(1, 1040, color='r', s=24)
    plt.subplot(1,3,3)

    TimeString = "Seconds into experiment that we reached 940g blood removed"
    data = df[TimeString].dropna()
    plt.title(TimeString, wrap = True)
    plt.boxplot(data[0:len(data)-1])
    plt.scatter(1,500, color = 'r', s = 24)
    plt.show()

if __name__ == "__main__":
    BleedingPlots()
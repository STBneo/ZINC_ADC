import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def CDS_histogram(fi):
    df = pd.read_csv(fi, sep='\t')
    df1 = df[df["Num_ZIDs"] <= 5000]
    df2 = df[df["Num_ZIDs"] > 5000]
    print(df)
    fig = plt.figure(figsize=(30, 30))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax1.hist(df1["Avg.BB_PCScore"], bins=20)
    ax2.hist(df1["Avg.BB_PCScore"], bins=20, cumulative=True)
    ax3.hist(df1["Num_ZIDs"], bins=20)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.set_title("Average BB PCScore", fontsize=20)
    ax2.set_title("Average BB PCScore Cumulative", fontsize=20)
    ax3.set_title("Num ZIDs", fontsize=20)

    ax1.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax1.set_yticks(np.arange(0, 450, 50))
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax3.tick_params(labelsize=15)

    plt.savefig("Samples.png", transparent=True)

    fig = plt.figure(figsize=(30, 30))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax1.hist(df2["Avg.BB_PCScore"], bins=20)
    ax2.hist(df2["Avg.BB_PCScore"], bins=20, cumulative=True)
    ax3.hist(df2["Num_ZIDs"], bins=10)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax1.set_title("Average BB PCScore", fontsize=20)
    ax2.set_title("Average BB PCScore Cumulative", fontsize=20)
    ax3.set_title("Num ZIDs", fontsize=20)

    ax1.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax3.tick_params(labelsize=15)

    plt.savefig("Over5000.png", transparent=True)
    return df1, df2


def CDS_histogram_each(fi):
    df = pd.read_csv(fi, sep='\t')
    df1 = df[df["Num_ZIDs"] <= 5000]
    df2 = df[df["Num_ZIDs"] > 5000]
    print(df)
    for j in ["Pass", "NotPass"]:
        if j == "Pass":
            pdf = df1
        else:
            pdf = df2
        for i in ["Avg.Score", "Avg.Score.Cumulative", "Num ZIDs"]:
            ptitle = ''
            if i == "Num ZIDs":
                ptitle = "Num ZIDs"
                plt.figure(figsize=(30, 15))
                coln = "Num_ZIDs"
            else:
                plt.figure(figsize=(15, 15))
                coln = "Avg.BB_PCScore"
                plt.xticks(np.arange(0.0, 1.1, 0.1))
            if i == "Avg.Score.Cumulative":
                ptitle = "Average BB PCScore Cumulative"
                plt.hist(pdf[coln], bins=20, cumulative=True)
            else:
                plt.hist(pdf[coln], bins=20)

            if ptitle == '':
                ptitle = "Average BB PCScore"
                if j == "Pass":
                    plt.yticks(np.arange(0, 450, 50))
                else:
                    pass
            #plt.title(ptitle, fontsize=20)
            plt.grid(True)
            #plt.show()
            plt.xticks(fontsize=45)
            plt.yticks(fontsize=45)
            plt.tight_layout()
            plt.savefig("%s.%s.png" % (j, i), transparent=True)
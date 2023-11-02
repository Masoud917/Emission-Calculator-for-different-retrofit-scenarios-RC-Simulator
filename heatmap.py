
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_heatmap(df, notb, output_filename):
    rows, columns = df.shape
    figsize = (columns * 2.5, rows * 1.5)

    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, cmap='YlGnBu', linewidths=0.5, fmt='.3f', center= True)
    
    plt.title(f'{notb} - Emission (KgCO2eq/m2) in 60 years')
    
    plt.tight_layout()
    plt.savefig(output_filename)
#-------------------------------------------------------------------------------
# Visualize labeled HCTSA dataset as a UMAP projection
#-------------------------------------------------------------------------------
import matplotlib as plt
import seaborn as sns
import pandas as pd
import umap
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# Use OutputToCSV to generate the following (default) files:
dataMatrixCSV = 'hctsa_datamatrix.csv'
timeSeriesInfoCSV = 'hctsa_timeseries-info.csv'

#-------------------------------------------------------------------------------
def LoadResults():
    # Load relevant hctsa results as a dataframe ():
    dataMatrix = pd.read_csv(dataMatrixCSV,header=None)
    tsLabels = pd.read_csv(timeSeriesInfoCSV,header=None,names=('name','label'))
    tsLabels['label'] = tsLabels['label'].astype('category')
    return dataMatrix,tsLabels

#-------------------------------------------------------------------------------
def UMAP_embed(dataMatrix):
    # Compute a 2d umap projection of the data in dataMatrix
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(dataMatrix)
    df = pd.DataFrame(data=embedding,columns=('umap-1','umap-2'))
    return df

#-------------------------------------------------------------------------------
def plot_projection(df,doSave=True):
    # Plot two-dimensional projection of data
    lowDim = sns.lmplot(x='umap-1', y='umap-2', data=df, fit_reg=False, markers='.',
                        hue='label', legend=True, legend_out=True, palette='Set2')
    # ax = plt.gca()
    # ax.set_title('UMAP projection of the hctsa dataset', fontsize=24);
    # ax.set_aspect('equal', 'datalim')
    if doSave:
        lowDim.savefig('umapProjection.pdf')
    return lowDim

#-------------------------------------------------------------------------------
def main():
    # Load data from hctsa calculation:
    dataMatrix,tsLabels = LoadResults()
    # Compute a umap projection in a dataframe:
    df_umap = UMAP_embed(dataMatrix)
    # Add label information:
    df_umap['label'] = tsLabels['label']
    # Plot
    plot_projection(df_umap)

if __name__ == "__main__": main()

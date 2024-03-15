# %matplotlib notebook

import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
import matplotlib
import pickle
import scipy
import time
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

# import tsne
# import sys; sys.path.append('/home/localadmin/github/FIt-SNE')
# from fast_tsne import fast_tsne
from my_t_sne import TSNE

# import my feature selection function and other helpful stuff
import rnaseqTools

def tasic_et_al_data_prep(init_technique='random', path=''):
    ########################### prepare ###########################

    def sns_styleset():
        sns.set_context('paper')
        sns.set_style('ticks')
        matplotlib.rcParams['axes.linewidth']    = .75
        matplotlib.rcParams['xtick.major.width'] = .75
        matplotlib.rcParams['ytick.major.width'] = .75
        matplotlib.rcParams['xtick.major.size'] = 3
        matplotlib.rcParams['ytick.major.size'] = 3
        matplotlib.rcParams['xtick.minor.size'] = 2
        matplotlib.rcParams['ytick.minor.size'] = 2
        matplotlib.rcParams['font.size']       = 7
        matplotlib.rcParams['axes.titlesize']  = 7
        matplotlib.rcParams['axes.labelsize']  = 7
        matplotlib.rcParams['legend.fontsize'] = 7
        matplotlib.rcParams['xtick.labelsize'] = 7
        matplotlib.rcParams['ytick.labelsize'] = 7
    #sns_styleset()

    ########################### download data ###########################
    #%%time

    filename = path+'mouse_VISp_2018-06-14_exon-matrix.csv'
    counts1, genes1, cells1 = rnaseqTools.sparseload(filename)
    filename = path+'mouse_ALM_2018-06-14_exon-matrix.csv'
    counts2, genes2, cells2 = rnaseqTools.sparseload(filename)
    counts = sparse.vstack((counts1, counts2), format='csc')
    cells = np.concatenate((cells1, cells2))

    if np.all(genes1==genes2):
        genes = np.copy(genes1)

    genesDF = pd.read_csv(path+'mouse_VISp_2018-06-14_genes-rows.csv')
    ids     = genesDF['gene_entrez_id'].tolist()
    symbols = genesDF['gene_symbol'].tolist()
    id2symbol = dict(zip(ids, symbols))
    genes = np.array([id2symbol[g] for g in genes])

    clusterInfo = pd.read_csv(path+'tasic-sample_heatmap_plot_data.csv')
    goodCells  = clusterInfo['sample_name'].values
    ids        = clusterInfo['cluster_id'].values
    labels     = clusterInfo['cluster_label'].values
    colors     = clusterInfo['cluster_color'].values

    clusterNames  = np.array([labels[ids==i+1][0] for i in range(np.max(ids))])
    clusterColors = np.array([colors[ids==i+1][0] for i in range(np.max(ids))])
    clusters   = np.copy(ids)

    ind = np.array([np.where(cells==c)[0][0] for c in goodCells])
    counts = counts[ind, :]

    areas = (ind < cells1.size).astype(int)

    clusters = clusters - 1

    tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
                 'clusterColors': clusterColors, 'clusterNames': clusterNames}

    # print(tasic2018['counts'].shape)
    # print(np.sum(tasic2018['areas']==0))
    # print(np.sum(tasic2018['areas']==1))
    # print(np.unique(tasic2018['clusters']).size)

    # pickle.dump(tasic2018, open('../data/tasic-nature/tasic2018.pickle', 'wb'))

    ########################### Feature selection ###########################

    markerGenes = ['Snap25','Gad1','Slc17a7','Pvalb', 'Sst', 'Vip', 'Aqp4',
               'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']

    # sns.set()
    importantGenesTasic2018 = rnaseqTools.geneSelection(
        tasic2018['counts'], n=3000, threshold=32, plot=False,
        markers=markerGenes, genes=tasic2018['genes'])
    # sns_styleset()


    ########################### Figure with tasic-et-al ###########################

    #%%time

    librarySizes = np.sum(tasic2018['counts'], axis=1)
    X = np.log2(tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6 + 1)
    X = np.array(X)
    X = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :50]

    C = tasic2018['clusterNames'].size
    clusterMeans = np.zeros((C, X.shape[1]))
    for c in range(C):
        clusterMeans[c, :] = np.mean(X[tasic2018['clusters'] == c, :], axis=0)

    if init_technique != 'random':
        pcaInit = X[:, :2] / np.std(X[:, 0]) * 0.0001

    return X, tasic2018


def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn

    cl, cl_inv = np.unique(classes, return_inverse=True)
    C = cl.size
    mu1 = np.zeros((C, X.shape[1]))
    mu2 = np.zeros((C, Z.shape[1]))
    for c in range(C):
        mu1[c, :] = np.mean(X[cl_inv == c, :], axis=0)
        mu2[c, :] = np.mean(Z[cl_inv == c, :], axis=0)

    nbrs1 = NearestNeighbors(n_neighbors=knn_classes).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)
    nbrs2 = NearestNeighbors(n_neighbors=knn_classes).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(C):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn_global = intersections / C / knn_classes

    subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
    d1 = pdist(X[subset, :])
    d2 = pdist(Z[subset, :])
    rho = scipy.stats.spearmanr(d1[:, None], d2[:, None]).correlation

    return (mnn, mnn_global, rho)



#
# # %matplotlib notebook
#
# import numpy as np
# import pylab as plt
# import seaborn as sns
# import pandas as pd
# import matplotlib
# import pickle
# import scipy
# import time
# from scipy import sparse
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import pdist
#
# # import tsne
# # import sys; sys.path.append('/home/localadmin/github/FIt-SNE')
# # from fast_tsne import fast_tsne
# from my_t_sne import TSNE
#
# # import my feature selection function and other helpful stuff
# import rnaseqTools
#
# def tasic_et_al_data_prep(init_technique='random'):
#     ########################### prepare ###########################
#
#     def sns_styleset():
#         sns.set_context('paper')
#         sns.set_style('ticks')
#         matplotlib.rcParams['axes.linewidth']    = .75
#         matplotlib.rcParams['xtick.major.width'] = .75
#         matplotlib.rcParams['ytick.major.width'] = .75
#         matplotlib.rcParams['xtick.major.size'] = 3
#         matplotlib.rcParams['ytick.major.size'] = 3
#         matplotlib.rcParams['xtick.minor.size'] = 2
#         matplotlib.rcParams['ytick.minor.size'] = 2
#         matplotlib.rcParams['font.size']       = 7
#         matplotlib.rcParams['axes.titlesize']  = 7
#         matplotlib.rcParams['axes.labelsize']  = 7
#         matplotlib.rcParams['legend.fontsize'] = 7
#         matplotlib.rcParams['xtick.labelsize'] = 7
#         matplotlib.rcParams['ytick.labelsize'] = 7
#     #sns_styleset()
#
#     ########################### download data ###########################
#     #%%time
#
#     filename = 'data/mouse_VISp_2018-06-14_exon-matrix.csv'
#     counts1, genes1, cells1 = rnaseqTools.sparseload(filename)
#     filename = 'data/mouse_ALM_2018-06-14_exon-matrix.csv'
#     counts2, genes2, cells2 = rnaseqTools.sparseload(filename)
#     counts = sparse.vstack((counts1, counts2), format='csc')
#     cells = np.concatenate((cells1, cells2))
#
#     if np.all(genes1==genes2):
#         genes = np.copy(genes1)
#
#     genesDF = pd.read_csv('data/mouse_VISp_2018-06-14_genes-rows.csv')
#     ids     = genesDF['gene_entrez_id'].tolist()
#     symbols = genesDF['gene_symbol'].tolist()
#     id2symbol = dict(zip(ids, symbols))
#     genes = np.array([id2symbol[g] for g in genes])
#
#     clusterInfo = pd.read_csv('data/tasic-sample_heatmap_plot_data.csv')
#     goodCells  = clusterInfo['sample_name'].values
#     ids        = clusterInfo['cluster_id'].values
#     labels     = clusterInfo['cluster_label'].values
#     colors     = clusterInfo['cluster_color'].values
#
#     clusterNames  = np.array([labels[ids==i+1][0] for i in range(np.max(ids))])
#     clusterColors = np.array([colors[ids==i+1][0] for i in range(np.max(ids))])
#     clusters   = np.copy(ids)
#
#     ind = np.array([np.where(cells==c)[0][0] for c in goodCells])
#     counts = counts[ind, :]
#
#     areas = (ind < cells1.size).astype(int)
#
#     clusters = clusters - 1
#
#     tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
#                  'clusterColors': clusterColors, 'clusterNames': clusterNames}
#
#     # print(tasic2018['counts'].shape)
#     # print(np.sum(tasic2018['areas']==0))
#     # print(np.sum(tasic2018['areas']==1))
#     # print(np.unique(tasic2018['clusters']).size)
#
#     # pickle.dump(tasic2018, open('../data/tasic-nature/tasic2018.pickle', 'wb'))
#
#     ########################### Feature selection ###########################
#
#     markerGenes = ['Snap25','Gad1','Slc17a7','Pvalb', 'Sst', 'Vip', 'Aqp4',
#                'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']
#
#     # sns.set()
#     importantGenesTasic2018 = rnaseqTools.geneSelection(
#         tasic2018['counts'], n=3000, threshold=32, plot=False,
#         markers=markerGenes, genes=tasic2018['genes'])
#     # sns_styleset()
#
#
#     ########################### Figure with tasic-et-al ###########################
#
#     #%%time
#
#     librarySizes = np.sum(tasic2018['counts'], axis=1)
#     X = np.log2(tasic2018['counts'][:, importantGenesTasic2018] / librarySizes * 1e+6 + 1)
#     X = np.array(X)
#     X = X - X.mean(axis=0)
#     U, s, V = np.linalg.svd(X, full_matrices=False)
#     U[:, np.sum(V, axis=1) < 0] *= -1
#     X = np.dot(U, np.diag(s))
#     X = X[:, np.argsort(s)[::-1]][:, :50]
#
#     C = tasic2018['clusterNames'].size
#     clusterMeans = np.zeros((C, X.shape[1]))
#     for c in range(C):
#         clusterMeans[c, :] = np.mean(X[tasic2018['clusters'] == c, :], axis=0)
#
#     if init_technique != 'random':
#         pcaInit = X[:, :2] / np.std(X[:, 0]) * 0.0001
#
#     return X
#
#
# def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
#     nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
#     ind1 = nbrs1.kneighbors(return_distance=False)
#
#     nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
#     ind2 = nbrs2.kneighbors(return_distance=False)
#
#     intersections = 0.0
#     for i in range(X.shape[0]):
#         intersections += len(set(ind1[i]) & set(ind2[i]))
#     mnn = intersections / X.shape[0] / knn
#
#     cl, cl_inv = np.unique(classes, return_inverse=True)
#     C = cl.size
#     mu1 = np.zeros((C, X.shape[1]))
#     mu2 = np.zeros((C, Z.shape[1]))
#     for c in range(C):
#         mu1[c, :] = np.mean(X[cl_inv == c, :], axis=0)
#         mu2[c, :] = np.mean(Z[cl_inv == c, :], axis=0)
#
#     nbrs1 = NearestNeighbors(n_neighbors=knn_classes).fit(mu1)
#     ind1 = nbrs1.kneighbors(return_distance=False)
#     nbrs2 = NearestNeighbors(n_neighbors=knn_classes).fit(mu2)
#     ind2 = nbrs2.kneighbors(return_distance=False)
#
#     intersections = 0.0
#     for i in range(C):
#         intersections += len(set(ind1[i]) & set(ind2[i]))
#     mnn_global = intersections / C / knn_classes
#
#     subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
#     d1 = pdist(X[subset, :])
#     d2 = pdist(Z[subset, :])
#     rho = scipy.stats.spearmanr(d1[:, None], d2[:, None]).correlation
#
#     return (mnn, mnn_global, rho)

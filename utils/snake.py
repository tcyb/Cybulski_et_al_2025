import glob
import gzip
import os

import pandas as pd
import scanpy as sc
import numpy as np
import scipy.io
import scipy.sparse

import sc_utils

def prepare_sample_mod(
    matrix,
    h5ad,
    meta,
    # markers,
    sample=None,
    min_genes=200,
    min_cells=5,
    sample_meta=None,
    doublets=None,
    library_col_name='Library ID'
):
    if os.path.exists(os.path.join(matrix, "matrix.mtx.gz")):
        try:
            ds = sc.read_10x_mtx(matrix, var_names="gene_symbols", make_unique=False)
        except KeyError:
            counts_matrix = scipy.io.mmread(gzip.open(
                os.path.join(matrix, "matrix.mtx.gz")
            )).T.tocsc()
            obs = pd.read_table(gzip.open(
                os.path.join(matrix, "barcodes.tsv.gz"
            )), header=None, index_col=0)
            obs.index.name = None
            var = pd.read_table(gzip.open(
                os.path.join(matrix, "features.tsv.gz"
            )), header=None).set_index(1)
            var.columns = ["Ensembl_ID"]
            var.index.name = None
            ds = sc.AnnData(counts_matrix, var=var, obs=obs)
    else:
        tenx_h5 = glob.glob(os.path.join(matrix, "*.h5"))
        if len(tenx_h5) == 1:
            tenx_h5 = tenx_h5[0]
        else:
            tenx_h5 = glob.glob(os.path.join(matrix, "filtered*.h5"))
            if len(tenx_h5) == 0:
                if os.path.exists(matrix) and matrix.endswith(".h5"):
                    tenx_h5 = matrix

        if len(tenx_h5) == 0:
            h5ad = glob.glob(os.path.join(matrix, "*.h5ad"))
            if len(h5ad) == 1:
                ds = sc.read_h5ad(h5ad[0])
            else:
                raise ValueError(f"Cannot find input in {matrix}")
        else:
            ds = sc.read_10x_h5(tenx_h5)
    ds.var_names_make_unique(join='.')
    ds.var_names = ds.var_names.str.replace('_', '-')

    ds.obs["orig.ident"] = sample
    
    if doublets:
        dbl_scores = pd.read_csv(doublets, index_col=0)
        dbl_scores.columns = ["cell", "doublet"]
        dbl_scores.set_index("cell", inplace=True)
        ds.obs["Doublet score"] = dbl_scores.doublet[ds.obs_names]
    
    ds.obs_names = sample + "_" + ds.obs_names.str.replace("-\d$", "")
    # scaled.columns = ds.obs_names

    sample_meta = pd.read_csv(sample_meta)
    this_meta = sample_meta.loc[sample_meta[library_col_name] == sample, :]
    #for _, v in this_meta.iteritems(): # deprecated in pandas 2.0
    for _, v in this_meta.items():
        ds.obs[v.name] = v.values[0]
    
    sc.pp.filter_cells(ds, min_genes=min_genes)
    sc.pp.filter_genes(ds, min_cells=min_cells) 
    
    ds.layers['counts'] = ds.X
    ds.raw = ds

    # defer mito/ribo to integration as well, as it causes issues with appending/joining
    # ds.var['mito'] = ds.var_names.str.startswith('MT-')
    # ds.var['ribo'] = ds.var_names.str.match("^RP(L|S)")
    # sc.pp.calculate_qc_metrics(
    #     ds,
    #     qc_vars=['mito', 'ribo'],
    #     percent_top=[10, 20],
    #     log1p=False,
    #     inplace=True
    # )

    # not sure if we need to zero-pad with missing_genes.
    #missing_genes = ds.var_names[~ds.var_names.isin(scaled.index)]
    #zeros = np.zeros((ds.n_obs, missing_genes.size))
    #zeros = pd.DataFrame(zeros, index=ds.obs_names, columns=missing_genes)
    #ds.X = pd.concat([scaled.T, zeros], axis=1).loc[:, ds.var_names]
    #ds.var["highly_variable"] = False
    #ds.var.loc[ds.var_names.isin(scaled.index), "highly_variable"] = True
    
    # skip PCA and leiden
    # defer rank gene groups and get_markers to scvi_integrate_mod 

    ds.write_h5ad(h5ad)
    ds.obs.to_csv(meta)


def prepare_sample(
    matrix,
    sct_counts,
    sct_hvg,
    h5ad,
    meta,
    markers,
    sample=None,
    min_genes=200,
    min_cells=5,
    n_pcs=40,
    resolution=0.5,
    sample_meta=None,
    doublets=None,
):
    if os.path.exists(os.path.join(matrix, "matrix.mtx.gz")):
        try:
            ds = sc.read_10x_mtx(matrix, var_names="gene_symbols", make_unique=False)
        except KeyError:
            counts_matrix = scipy.io.mmread(gzip.open(
                os.path.join(matrix, "matrix.mtx.gz")
            )).T.tocsc()
            obs = pd.read_table(gzip.open(
                os.path.join(matrix, "barcodes.tsv.gz"
            )), header=None, index_col=0)
            obs.index.name = None
            var = pd.read_table(gzip.open(
                os.path.join(matrix, "features.tsv.gz"
            )), header=None).set_index(1)
            var.columns = ["Ensembl_ID"]
            var.index.name = None
            ds = sc.AnnData(counts_matrix, var=var, obs=obs)
    else:
        tenx_h5 = glob.glob(os.path.join(matrix, "*.h5"))
        if len(tenx_h5) == 1:
            tenx_h5 = tenx_h5[0]
        else:
            tenx_h5 = glob.glob(os.path.join(matrix, "filtered*.h5"))
            if len(tenx_h5) == 0:
                if os.path.exists(matrix) and matrix.endswith(".h5"):
                    tenx_h5 = matrix

        if len(tenx_h5) == 0:
            h5ad = glob.glob(os.path.join(matrix, "*.h5ad"))
            if len(h5ad) == 1:
                ds = sc.read_h5ad(h5ad[0])
            else:
                raise ValueError(f"Cannot find input in {matrix}")
        else:
            ds = sc.read_10x_h5(tenx_h5)
    ds.var_names_make_unique(join=".")
    ds.var_names = ds.var_names.str.replace("_", "-")

    scaled = pd.read_csv(sct_hvg, sep="\t")
    ds = ds[scaled.columns, :].copy()

    ds.obs["orig.ident"] = sample
    if doublets:
        dbl_scores = pd.read_csv(doublets, index_col=0)
        dbl_scores.columns = ["cell", "doublet"]
        dbl_scores.set_index("cell", inplace=True)
        ds.obs["Doublet score"] = dbl_scores.doublet[ds.obs_names]
    ds.obs_names = sample + "_" + ds.obs_names.str.replace("-\d$", "")
    scaled.columns = ds.obs_names

    sample_meta = pd.read_csv(sample_meta)
    this_meta = sample_meta.loc[sample_meta["External Sample ID"] == sample, :]
    for _, v in this_meta.iteritems():
        ds.obs[v.name] = v.values[0]

    sc.pp.filter_cells(ds, min_genes=min_genes)
    sc.pp.filter_genes(ds, min_cells=min_cells)
    ds.layers["counts"] = ds.X

    ds.var["mito"] = ds.var_names.str.startswith("MT-")
    ds.var["ribo"] = ds.var_names.str.match("^RP(L|S)")
    sc.pp.calculate_qc_metrics(
        ds,
        qc_vars=["mito", "ribo"],
        percent_top=[10, 20],
        log1p=False,
        inplace=True
    )

    counts = scipy.io.mmread(sct_counts)
    ds.X = counts.tocsc().T
    ds.raw = ds

    missing_genes = ds.var_names[~ds.var_names.isin(scaled.index)]
    zeros = np.zeros((ds.n_obs, missing_genes.size))
    zeros = pd.DataFrame(zeros, index=ds.obs_names, columns=missing_genes)
    ds.X = pd.concat([scaled.T, zeros], axis=1).loc[:, ds.var_names]
    ds.var["highly_variable"] = False
    ds.var.loc[ds.var_names.isin(scaled.index), "highly_variable"] = True

    sc.tl.pca(ds, svd_solver="arpack", use_highly_variable=True)
    sc.pp.neighbors(ds, n_pcs=n_pcs)
    sc.tl.leiden(ds, resolution=resolution)
    sc.tl.umap(ds)

    ds.write_h5ad(h5ad)
    ds.obs.to_csv(meta)

    sc.tl.rank_genes_groups(ds, "leiden", method="wilcoxon", n_genes=200)
    m = sc_utils.get_markers(ds, "leiden")
    m.to_csv(markers)


def split_demux(sample_dir, sample_clusters, sample, output_dir):
    hto = sample.split("+")[1]
    in_sample = sample.split("+")[0]
    ds = sc.read_10x_h5(os.path.join(sample_dir, f"{in_sample}.h5"))
    clusters = pd.read_csv(sample_clusters, index_col=0)
    ds = ds[ds.obs_names.isin(clusters.index), :]
    ds.obs["cluster"] = clusters.x[ds.obs_names]
    clusters = pd.Series(ds.obs.cluster.unique())
    hto_cluster = clusters[clusters.str.contains(hto)].values[0]
    ds = ds[ds.obs.cluster == hto_cluster, :]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd.DataFrame({
        0: ds.var.index,
        1: ds.var.index,
        2: "Gene Expression"
    }).to_csv(
        os.path.join(output_dir, "features.tsv.gz"),
        sep="\t",
        index=False,
        header=False
    )
    pd.DataFrame(ds.obs.index).to_csv(
        os.path.join(output_dir, "barcodes.tsv.gz"),
        sep="\t",
        index=False,
        header=False
    )
    with gzip.open(os.path.join(output_dir, "matrix.mtx.gz"), "wb") as f:
        scipy.io.mmwrite(f, ds.X.T)


def merge(h5ads, output_dir):
    datasets = []
    for h5ad in h5ads:
        ds = sc.read_h5ad(h5ad)
        ds.X = ds.layers["counts"]
        datasets.append(ds)

    ds = datasets[0].concatenate(datasets[1:], join="outer")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sc_utils.write_mtx(ds, output_dir)
    
def merge_mod(h5ads, out):
    datasets = []
    for h5ad in h5ads:
        ds = sc.read_h5ad(h5ad)
        ds.X = ds.layers["counts"]
        datasets.append(ds)

    ds = datasets[0].concatenate(datasets[1:], join="outer")
    
    if out.endswith('.h5ad'):
        ds.write_h5ad(out)
        return
    if not os.path.exists(out):
        os.makedirs(output_dir)
    sc_utils.write_mtx(ds, out)


def filter_merge(h5ads, output_dir, samples=None, exclude=None, include=None):
    sample_meta = pd.read_csv(samples)

    datasets = []
    for h5ad in h5ads:
        ds = sc.read_h5ad(h5ad)
        ds.X = ds.layers["counts"]
        sample = ds.obs["orig.ident"].values[0]
        doublet_threshold = sample_meta["Doublet threshold"][sample_meta["External Sample ID"] == sample].values[0]
        ds = ds[ds.obs["Doublet score"] < doublet_threshold, :]
        datasets.append(ds)

    ds = datasets[0].concatenate(datasets[1:], join="outer")
    if exclude:
        exclude = pd.read_csv(exclude, header=None)
        ds = ds[~ds.obs_names.isin(exclude.iloc[:, 0]), :].copy()
    if include:
        include = pd.read_csv(include, index_col=0)
        ds = ds[ds.obs_names.isin(include.iloc[:, 0]), :].copy()

    if output_dir.endswith(".h5ad"):
        ds.write_h5ad(output_dir)
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sc_utils.write_mtx(ds, output_dir)

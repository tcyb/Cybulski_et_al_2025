import os

import numpy as np
import pandas as pd
import scanpy as sc
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import bbknn
import scipy.io
import scipy.sparse
import scvi

import sc_utils




def integrate(
    matrix,
    sct_counts,
    sct_hvg,
    h5ad,
    meta,
    markers,
    sample_meta=None,
    min_genes=200,
    min_cells=5,
    n_pcs=40,
    resolution=0.5,
    neighbors=10,
    ignore_meta=None,
):
    if ignore_meta is None:
        ignore_meta = []
    ds = sc.read_10x_mtx(matrix, var_names="gene_symbols", make_unique=False)
    ds.var_names = ds.var_names.str.replace("_", "-")

    scaled = pd.read_csv(sct_hvg, sep="\t")
    ds = ds[scaled.columns, :].copy()

    sample_meta = pd.read_csv(sample_meta)
    for sample in sample_meta["External Sample ID"]:
        this_meta = sample_meta.loc[sample_meta["External Sample ID"] == sample, :]
        for _, v in this_meta.iteritems():
            if v.name in ignore_meta:
                continue
            ds.obs.loc[ds.obs_names.str.startswith(f"{sample}_"), v.name] = v.values[0]
    ds.obs["orig.ident"] = ds.obs["External Sample ID"]

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

    ds = ds[:, ds.var_names.isin(scaled.index)]
    ds.var["highly_variable"] = True
    ds.X = scaled.T

    sc.tl.pca(ds, svd_solver="arpack", use_highly_variable=True)
    bbknn.bbknn(ds, neighbors_within_batch=neighbors, n_pcs=n_pcs, batch_key="orig.ident")
    sc.tl.leiden(ds, resolution=resolution)
    sc.tl.umap(ds)

    ds.write_h5ad(h5ad)
    ds.obs.to_csv(meta)

    sc.tl.rank_genes_groups(ds, "leiden", method="wilcoxon", n_genes=200)
    m = sc_utils.get_markers(ds, "leiden")
    m.to_csv(markers)


def integrate_scvi(
    input_h5ad,
    h5ad,
    meta,
    markers,
    batch_key=None,
    n_latent=50,
    n_epochs=400,
    model_path=None,
    resolution=0.5,
    scvi_kwargs=None,
    n_hvg=None
):
    # For GPU on quest
    # https://github.com/ray-project/ray/issues/10995#issuecomment-698177711
    os.environ["SLURM_JOB_NAME"] = "bash"
    
    if batch_key is None:
        raise ValueError("Please provide a batch_key")

    if scvi_kwargs is None:
        scvi_kwargs = {}

    ds = sc.read_h5ad(input_h5ad)
    ds.obs[batch_key] = ds.obs[batch_key].astype("str")
    if n_hvg is not None:
        sc.pp.highly_variable_genes(
            ds,
            n_top_genes=n_hvg,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            batch_key=batch_key,
        )
    scvi.data.setup_anndata(ds, layer="counts", batch_key=batch_key)
    vae = scvi.model.SCVI(ds, n_latent=n_latent, **scvi_kwargs)
    vae.train(max_epochs=n_epochs)
    if model_path:
        vae.save(model_path, overwrite=True)
    ds.obsm["X_scVI"] = vae.get_latent_representation()
    sc.pp.neighbors(ds, use_rep="X_scVI")
    sc.tl.leiden(ds, resolution=resolution)
    sc.tl.umap(ds)
    sc.tl.rank_genes_groups(ds, "leiden", method="wilcoxon", n_genes=200)
    m = sc_utils.get_markers(ds, "leiden")
    ds.write_h5ad(h5ad)
    ds.obs.to_csv(meta)
    m.to_csv(markers)
    

def integrate_scvi_mod(
    input_h5ad,
    h5ad,
    # meta,
    # markers,
    batch_key=None,
    min_genes=200,
    min_cells=5,
    n_latent=50,
    n_epochs=400,
    model_path=None,
    resolution=0.5,
    scvi_kwargs=None,
    n_hvg=None
):
    # For GPU on quest
    # https://github.com/ray-project/ray/issues/10995#issuecomment-698177711
    os.environ["SLURM_JOB_NAME"] = "bash"
    
    if batch_key is None:
        raise ValueError("Please provide a batch_key")

    if scvi_kwargs is None:
        scvi_kwargs = {}

    ds = sc.read_h5ad(input_h5ad)
    ds.obs[batch_key] = ds.obs[batch_key].astype("str")
    
    # should already be filtered on a per-subject level
    sc.pp.filter_cells(ds, min_genes=min_genes)
    sc.pp.filter_genes(ds, min_cells=min_cells)
    
    # store filtered counts in 'counts'
    ds.layers['counts'] = ds.X
    ds.raw = ds
    
    if n_hvg is not None:
        sc.pp.highly_variable_genes(
            ds,
            n_top_genes=n_hvg,
            subset=True,
            layer="counts",
            flavor="seurat_v3",
            batch_key=batch_key,
        )
    
    ds.var['mito'] = ds.var_names.str.startswith('MT-')
    ds.var['ribo'] = ds.var_names.str.match("^RP(L|S)")
    sc.pp.calculate_qc_metrics(
        ds,
        qc_vars=['mito', 'ribo'],
        percent_top=[10, 20],
        log1p=False,
        inplace=True
    )
    
    ds.var.rename(columns={'gene_ids': 'ensembl_ids'}, inplace=True)
    ds.raw.var.rename(columns={'gene_ids': 'ensembl_ids'}, inplace=True)
    
    scvi.model.SCVI.setup_anndata(ds, layer="counts", batch_key=batch_key)
    vae = scvi.model.SCVI(ds, n_latent=n_latent, **scvi_kwargs)
    vae.train(max_epochs=n_epochs)
    if model_path:
        vae.save(model_path, overwrite=True)
    ds.obsm["X_scVI"] = vae.get_latent_representation()
    sc.pp.neighbors(ds, use_rep="X_scVI")
    sc.tl.leiden(ds, key_added='leiden', resolution=resolution)
    sc.tl.umap(ds)
    ds.write_h5ad(h5ad) # taking markers and meta out of this function and into its own
    
    
def generate_markers_and_meta(input_h5ad, markers_fname, meta_fname,
                              n_genes=200,
                              use_raw=False,
                              clust_name='leiden'):
    ds = sc.read_h5ad(input_h5ad)
    # normalize and log-transform
    sc.pp.normalize_total(ds, target_sum=1e4)
    sc.pp.log1p(ds)
   
    sc.tl.rank_genes_groups(ds, clust_name, method="wilcoxon", # takes normalized/log-transformed data
                            n_genes=n_genes, use_raw=use_raw)
    m = sc_utils.get_markers(ds, clust_name)
    ds.obs.to_csv(meta_fname)
    m.to_csv(markers_fname)
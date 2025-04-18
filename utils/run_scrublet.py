import glob
import gzip
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scipy.io
import scrublet as scr


def run_scrublet_mod(tenx_dir, doublet_rate=0.06, npca=40,
                     save_csv_to=None, save_hist_to=None, save_threshold_to=None,
                     barcodes=None):
    if not save_csv_to:
        raise ValueError("Please, specify prefix path where to save results to")

    if os.path.exists(os.path.join(tenx_dir, "outs")): # Raw cellranger output
        if barcodes is None:
            tenx_dir = os.path.join(tenx_dir, "outs", "filtered_feature_bc_matrix")
            counts_matrix = scipy.io.mmread(gzip.open(
                os.path.join(tenx_dir, "matrix.mtx.gz")
            )).T.tocsc()
            obs = pd.read_table(gzip.open(
                os.path.join(tenx_dir, "barcodes.tsv.gz"
            )), header=None)
        else:
            tenx_dir = os.path.join(tenx_dir, "outs", "raw_feature_bc_matrix")
            counts_matrix = scipy.io.mmread(gzip.open(
                os.path.join(tenx_dir, "matrix.mtx.gz")
            )).T.tocsc()
            obs = pd.read_table(gzip.open(
                os.path.join(tenx_dir, "barcodes.tsv.gz"
            )), header=None)
            bc = pd.read_csv(barcodes, header=None)
            counts_matrix = counts_matrix[obs["0"].isin(bc[0]), :]
            obs = obs.loc[obs["0"].isin(bc[0]), :]
    elif os.path.exists(os.path.join(tenx_dir, "matrix.mtx.gz")):
        counts_matrix = scipy.io.mmread(gzip.open(
            os.path.join(tenx_dir, "matrix.mtx.gz")
        )).T.tocsc()
        obs = pd.read_table(gzip.open(
            os.path.join(tenx_dir, "barcodes.tsv.gz"
        )), header=None)
    else:
        # Possible options: h5 file or dir with h5 files
        # Or: dir with h5ad file
        tenx_h5 = glob.glob(os.path.join(tenx_dir, "*.h5"))
        if len(tenx_h5) == 1:
            tenx_h5 = tenx_h5[0]
        elif tenx_dir.endswith(".h5"):
            tenx_h5 = tenx_dir
        else:
            if barcodes is None:
                tenx_h5 = glob.glob(os.path.join(tenx_dir, "filtered*.h5"))
            else:
                tenx_h5 = glob.glob(os.path.join(tenx_dir, "raw*.h5"))

        if len(tenx_h5) == 0:
            h5ad = glob.glob(os.path.join(tenx_dir, "*.h5ad"))
            if len(h5ad) == 1:
                ds = sc.read_h5ad(h5ad[0])
                counts_matrix = ds.X.tocsc().astype(np.longlong)
                obs = ds.obs.reset_index().iloc[:, [0]]
                obs.columns = ["0"]
            else:
                raise ValueError(f"Cannot find input in {tenx_dir}")
        else:
            ds = sc.read_10x_h5(tenx_h5[0])
            if barcodes is not None:
                bc = pd.read_csv(barcodes, header=None)
                ds = ds[bc[0], :]
            counts_matrix = ds.X.tocsc().astype(np.longlong)
            obs = ds.obs.reset_index()
            obs.columns = ["0"]

    row_sums = (np.sum(counts_matrix, axis=1) > 20).A1
    counts_matrix = counts_matrix[row_sums, :]
    obs = obs.loc[row_sums, :]

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=doublet_rate)
    doublet_scores, doublets = scrub.scrub_doublets(min_counts=2,
                                                    min_cells=3,
                                                    min_gene_variability_pctl=85,
                                                    n_prin_comps=npca)
    if doublets is None:
        scrub.call_doublets(threshold=0.4)
    # save_dir = os.path.dirname(save_to)
    # if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    obs["doublet"] = doublet_scores
    obs.to_csv(save_csv_to)
    scrub.plot_histogram()
    plt.savefig(save_hist_to)
    if not os.path.exists(save_threshold_to):
        with open(save_threshold_to, 'w') as f:
            f.write(str(scrub.threshold_))


def run_scrublet(tenx_dir, doublet_rate=0.06, npca=40, save_to=None, barcodes=None):
    if not save_to:
        raise ValueError("Please, specify prefix path where to save results to")

    if os.path.exists(os.path.join(tenx_dir, "outs")): # Raw cellranger output
        if barcodes is None:
            tenx_dir = os.path.join(tenx_dir, "outs", "filtered_feature_bc_matrix")
            counts_matrix = scipy.io.mmread(gzip.open(
                os.path.join(tenx_dir, "matrix.mtx.gz")
            )).T.tocsc()
            obs = pd.read_table(gzip.open(
                os.path.join(tenx_dir, "barcodes.tsv.gz"
            )), header=None)
        else:
            tenx_dir = os.path.join(tenx_dir, "outs", "raw_feature_bc_matrix")
            counts_matrix = scipy.io.mmread(gzip.open(
                os.path.join(tenx_dir, "matrix.mtx.gz")
            )).T.tocsc()
            obs = pd.read_table(gzip.open(
                os.path.join(tenx_dir, "barcodes.tsv.gz"
            )), header=None)
            bc = pd.read_csv(barcodes, header=None)
            counts_matrix = counts_matrix[obs["0"].isin(bc[0]), :]
            obs = obs.loc[obs["0"].isin(bc[0]), :]
    elif os.path.exists(os.path.join(tenx_dir, "matrix.mtx.gz")):
        counts_matrix = scipy.io.mmread(gzip.open(
            os.path.join(tenx_dir, "matrix.mtx.gz")
        )).T.tocsc()
        obs = pd.read_table(gzip.open(
            os.path.join(tenx_dir, "barcodes.tsv.gz"
        )), header=None)
    else:
        # Possible options: h5 file or dir with h5 files
        # Or: dir with h5ad file
        tenx_h5 = glob.glob(os.path.join(tenx_dir, "*.h5"))
        if len(tenx_h5) == 1:
            tenx_h5 = tenx_h5[0]
        elif tenx_dir.endswith(".h5"):
            tenx_h5 = tenx_dir
        else:
            if barcodes is None:
                tenx_h5 = glob.glob(os.path.join(tenx_dir, "filtered*.h5"))
            else:
                tenx_h5 = glob.glob(os.path.join(tenx_dir, "raw*.h5"))

        if len(tenx_h5) == 0:
            h5ad = glob.glob(os.path.join(tenx_dir, "*.h5ad"))
            if len(h5ad) == 1:
                ds = sc.read_h5ad(h5ad[0])
                counts_matrix = ds.X.tocsc().astype(np.longlong)
                obs = ds.obs.reset_index().iloc[:, [0]]
                obs.columns = ["0"]
            else:
                raise ValueError(f"Cannot find input in {tenx_dir}")
        else:
            ds = sc.read_10x_h5(tenx_h5[0])
            if barcodes is not None:
                bc = pd.read_csv(barcodes, header=None)
                ds = ds[bc[0], :]
            counts_matrix = ds.X.tocsc().astype(np.longlong)
            obs = ds.obs.reset_index()
            obs.columns = ["0"]

    row_sums = (np.sum(counts_matrix, axis=1) > 20).A1
    counts_matrix = counts_matrix[row_sums, :]
    obs = obs.loc[row_sums, :]

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=doublet_rate)
    doublet_scores, doublets = scrub.scrub_doublets(min_counts=2,
                                                    min_cells=3,
                                                    min_gene_variability_pctl=85,
                                                    n_prin_comps=npca)
    if doublets is None:
        scrub.call_doublets(threshold=0.4)
    save_dir = os.path.dirname(save_to)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    obs["doublet"] = doublet_scores
    obs.to_csv(save_to + "doublets.csv")
    scrub.plot_histogram()
    plt.savefig(save_to + 'doublet_hist.pdf')
    if not os.path.exists(save_to + 'threshold.txt'):
        with open(save_to + 'threshold.txt', 'w') as f:
            f.write(str(scrub.threshold_))
from collections import defaultdict
import math
import openpyxl
import os
import re

import numpy as np
import pandas as pd
import scipy.stats


def load_annotations():
    wb = openpyxl.load_workbook("../00annotation.xlsx")
    populations = wb.get_sheet_by_name("Samples populations")
    ds = pd.DataFrame(populations.values)
    ds.columns = ds.iloc[0, :]
    ds = ds.iloc[1:, :].reset_index(drop=True)
    ds = ds.loc[~ds.iloc[:, 1].isna(), :]
    inserted = 0
    for col in range(2, ds.shape[1], 2):
        comment_col = []
        for i, row in enumerate(populations.iter_rows()):
            if i == 0:
                continue
            if i > ds.shape[0]:
                break
            cell = row[col]
            comment_col.append(cell.comment.text if cell.comment else None)
        ds.insert(1 + col + inserted, f"{col}_comment", comment_col)
        inserted += 1
    return ds


def load_ontology():
    ontology = pd.read_excel(
        "../00annotation.xlsx",
        sheet_name="Cell type labels",
        engine="openpyxl"
    )

    ontology_dict = {}
    parents = {}
    parents[0] = None
    for x in ontology.itertuples(name=None, index=False):
        for i, cell in enumerate(x):
            if cell in [np.nan, ">", "|", None]:
                continue
            if cell in ontology_dict:
                cell = f"{cell}-v2"
            short_name = re.search(r"\(.+\)$", cell)
            if short_name:
                short_name = short_name.group(0)[1:-1]
                cell = cell[:-len(short_name) - 2]
                cell = cell.strip()
            else:
                short_name = cell
            parent = parents[i - 1]
            ontology_dict[cell.lower()] = [cell, i, parent, short_name]
            parents[i] = cell
    return ontology_dict


def load_samples_meta():
    return pd.read_csv("../00all-samples.csv")


def load_manual():
    manual = open("../08integration/manual.txt", "rt").read().replace("\\\\", "\\")
    manual = [x.split("|") for x in manual.split("\n")]
    return manual


def load_mapping():
    mapping = open("../08integration/mapping.txt", "rt").read()
    mapping = {x.split(",")[0]: x.split(",")[1] for x in mapping.strip().split("\n")}
    return mapping


def load_objects_meta(samples_meta):
    imeta = {}
    for x in samples_meta.itertuples():
        sample = x._4
        dir_ = x.Directory
        meta_path = os.path.join("../../data", dir_, "scanpy", sample, f"{sample}-metadata.csv")
        imeta[sample] = pd.read_csv(meta_path, index_col=0)
    return imeta


def clear_comment(text):
    text = text.replace("-Nikolay Markov", "")
    text = text.replace("-Alexander Misharin", "")
    return text.strip()


def find_cell_type(ct, mapping, ontology_dict):
    ct = ct.lower()
    for orig, repl in mapping.items():
        if orig.lower() in ct:
            ct = repl.lower()
            break
    result = ontology_dict.get(ct, None)
    if result is None:
        x = [i for i in ontology_dict.values() if i[3].lower() == ct]
        if len(x) == 1:
            result = x[0]
    return result


def find_cell_types(ct, comment, manual, mapping, ontology_dict):
    orig_ct = ct
    if ct == "Fibroblasts":
        ct = "Fibroblast"
    if " and " in ct:
        ct = ct.split(" and ")
    elif " + " in ct:
        ct = ct.split(" + ")
    else:
        ct = [ct]
    if comment.lower().startswith("and "):
        ct += re.split("(?:,? and )|(?:, )", comment[4:].lower())
    if comment.startswith("+ "):
        ct += re.split("(?:,? and )|(?:, )", comment[2:].lower())
    if comment.lower().startswith("with "):
        ct.append(comment[5:])
    if " + MT AM" in comment:
        ct.append("MT AM")
    if orig_ct == "FABP4 AM" and "IFI27" in comment:
        ct = ["IFI27 AM"]
    if orig_ct == 'Monocytes' and comment.startswith('Both classical and non-classical'):
        ct = ["cMo", "ncMo"]
    for x in manual:
        if orig_ct == x[0] and comment.replace("\n", "\\n") == x[1]:
            ct = x[2].split(",")
    return [find_cell_type(x, mapping, ontology_dict) for x in ct]


def set_cell_types(row, table, manual, mapping, ontology_dict):
    t = {}
    sample = row[0]
    not_found = []
    for x in range(1, len(row), 3):
        cell_count = row[x]
        cell_type = row[x + 1]
        if x + 2 < len(row):
            comment = row[x + 2]
        else:
            comment = ""
        if type(cell_type) is not str:
            break
        if type(comment) is not str:
            comment = ""
        comment = clear_comment(comment)
        cluster = str(x//3)
        ont_ct = find_cell_types(cell_type, comment, manual, mapping, ontology_dict)
        t[cluster] = (cell_type, comment, ont_ct)
        if any([x is None for x in ont_ct]):
            raise ValueError(f"{cell_type} | {comment} => {ont_ct}")
    table[sample] = t


def get_ct(cell_types, level, ontology_dict):
    result = set()
    for ct in cell_types:
        while ct[1] > level:
            ct = ontology_dict[ct[2].lower()]
        result.add(ct[3])
    return ",".join(sorted(result))


def annotate(input_meta, output, entropy_groups=["leiden"]):
    ontology_dict = load_ontology()
    annotations = load_annotations()
    samples_meta = load_samples_meta()
    objects_meta = load_objects_meta(samples_meta)

    manual = load_manual()
    mapping = load_mapping()

    meta = pd.read_csv(input_meta, index_col=0)
    big_cell_names = meta.index.str.replace("-\d+$", "")

    for i in range(1, 6):
        meta[f"cell_type_{i}"] = ""
        for group in entropy_groups:
            meta[f"entropy_{group}_{i}"] = np.nan
    meta["comment"] = ""

    table = {}
    for row in annotations.itertuples(index=False, name=None):
        if type(row[0]) != str:
            break
        set_cell_types(row, table, manual, mapping, ontology_dict)

    for sample, sample_meta in objects_meta.items():
        assignments = table[sample]
        for ct, info in assignments.items():
            cells = sample_meta.index[sample_meta.leiden == int(ct)]
            comment = info[1]
            meta.loc[big_cell_names.isin(cells), "comment"] = comment
            for i in range(5, 0, -1):
                ct = get_ct(info[2], i, ontology_dict)
                meta.loc[big_cell_names.isin(cells), f"cell_type_{i}"] = ct

    for group in entropy_groups:
        for cl in meta[group].unique():
            cells = meta.index[meta[group] == cl]
            for i in range(1, 6):
                ct = meta.loc[cells, f"cell_type_{i}"].value_counts()
                ct_counts = defaultdict(int)
                for cell_types, count in ct.to_dict().items():
                    cell_types = cell_types.split(",")
                    for cell_type in cell_types:
                        ct_counts[cell_type] += count / len(cell_types)
                ct_dist = [x / sum(ct_counts.values()) for x in ct_counts.values()]
                entropy_value = scipy.stats.entropy(ct_dist, base=2)
                meta.loc[cells, f"entropy_{group}_{i}"] = entropy_value

    meta.drop(["n_genes_by_counts", "total_counts_mito", "total_counts_ribo"], axis=1, inplace=True)
    meta.comment = meta.comment.str.replace("\n", " ")
    meta.to_csv(output)

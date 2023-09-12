"""
fach.py

Generate fatty acid composition heatmaps from a table of area values obtained by
untargeted lipidomics.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import random
import re
import seaborn as sns
from tqdm.auto import tqdm

plt.rcParams["font.family"] = "Arial"


def parse_lipid_annotation(l):
    """
    Parse a lipid annotation into its corresponding lipid class, number of
    carbon atoms, and number of double bonds.

    :param l:   lipid annotation to be parsed

    :return:    list containing [lipid class, n_c, n_db]
    """
    pattern = "^[A-z]+ [O-]*[P-]*\d+:\d+"
    # Expected pattern not found, skip parsing this lipid annotation
    if len(re.findall(pattern, l)) == 0:
        return None
    annotation = re.findall(pattern, l)
    assert len(annotation) == 1
    annotation = annotation[0]
    if "P-" in annotation or "O-" in annotation:
        lipid_class = annotation.split("-")[0] + "-"
        composition = annotation.split("-")[1]
    else:
        lipid_class = annotation.split(" ")[0]
        composition = annotation.split(" ")[1]
    [n_c, n_db] = composition.split(":")
    return [lipid_class, n_c, n_db]


def pad_margin(area_df, margin):
    assert margin in ["N_Carbon", "N_DB"]
    # Determine min and max values
    margin_range = np.arange(area_df[margin].min(), area_df[margin].max() + 1, 1)
    area_df = (
        pd.DataFrame({margin: [margin_range]})
        .explode(margin)
        .merge(area_df, on=[margin], how="outer")
        .astype({margin: "int32"})
    )
    return area_df


def plot_fach(area_df, mean_markers, heatmap_cmap):
    """
    Plot a fatty acid composition heatmap.

    :param area_df:         a pandas DataFrame holding N_Carbon/N_DB data for
                            features within a lipid class and sample group

    :param mean_markers:    a boolean denoting whether or not to draw markers
                            showing mean number of carbon atoms/double bonds

    :param heatmap_cmap:    a string denoting the desired heatmap colourmap

    :return:                a Matplotlib figure
    """
    # Create wide matrix for sns.heatmap
    heatmap_df = area_df.pivot(columns="N_Carbon", index="N_DB", values="Proportion")
    heatmap_df.fillna(0, inplace=True)
    # Initalize a grid of subplots
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.1,
        hspace=0.1,
    )
    ax_heatmap = fig.add_subplot(
        gs[1, 0],
    )
    ax_hist_x = fig.add_subplot(gs[0, 0])
    ax_hist_y = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[0, 1])
    # Add plots to the subplot axes
    sns.heatmap(
        data=heatmap_df,
        ax=ax_heatmap,
        cbar_ax=ax_cbar,
        cmap=heatmap_cmap,
        cbar_kws={"orientation": "horizontal", "label": "Proportion"},
    )
    sns.barplot(
        data=(area_df.groupby("N_Carbon", as_index=False).sum("Proportion")),
        x="N_Carbon",
        y="Proportion",
        color="grey",
        errorbar=None,
        ax=ax_hist_x,
        width=0.9,
    )
    sns.barplot(
        data=(area_df.groupby("N_DB", as_index=False).sum("Proportion")),
        y="N_DB",
        x="Proportion",
        orient="h",
        color="grey",
        errorbar=None,
        ax=ax_hist_y,
        width=0.9,
    )
    # Determine means and mark them on the heatmap if the flag is set
    if mean_markers:
        n_carbon_range = np.arange(
            area_df["N_Carbon"].min(), area_df["N_Carbon"].max(), 1
        )
        n_db_range = np.arange(area_df["N_DB"].min(), area_df["N_DB"].max(), 1)
        avg_n_carbon = sum(area_df["N_Carbon"] * area_df["Proportion"])
        avg_n_db = sum(area_df["N_DB"] * area_df["Proportion"])
        if len(n_carbon_range) > 1:
            ax_heatmap.axvline(
                x=np.interp(avg_n_carbon, n_carbon_range, range(len(n_carbon_range))) + 0.5,
                linestyle="--",
                linewidth=1,
            )
        if len(n_db_range) > 1:
            ax_heatmap.axhline(
                y=np.interp(avg_n_db, n_db_range, range(len(n_db_range))) + 0.5,
                linestyle="--",
                linewidth=1,
            )
    # Decorating plots
    ax_hist_x.spines[["right", "top"]].set_visible(False)
    ax_hist_y.spines[["right", "top"]].set_visible(False)
    ax_hist_x.tick_params(labelbottom=False, bottom=False, size=12)
    ax_hist_y.tick_params(labelleft=False, left=False, size=12)
    ax_hist_x.set_xlabel(None)
    ax_hist_y.set_ylabel(None)
    ax_heatmap.set_xlabel("Number of carbon atoms")
    ax_heatmap.set_ylabel("Number of double bonds")
    ax_cbar.xaxis.set_ticks_position("top")
    return fig


def plot_marginal_barplot(area_df, margin, pad_values=False):
    assert margin in ["N_Carbon", "N_DB"]
    # Sum proportions if they share the same N_Carbon or N_DB, depending on the
    # marginal varable to be plotted
    area_df = (
        area_df.sort_values(by=[margin, "Sample_Group"])
        .groupby([margin, "Sample_Group"])
        .sum()["Proportion"]
        .reset_index()
    )
    if pad_values:
        area_df = pad_margin(area_df, margin)
    p_aspect = 2.5 if area_df[margin].max() - area_df[margin].min() <= 10 else 4
    margin_fontsize = 12 if area_df[margin].max() - area_df[margin].min() <= 10 else 8
    p = sns.catplot(
        kind="bar",
        data=area_df,
        x=margin,
        y="Proportion",
        col_wrap=2,
        col="Sample_Group",
        height=1,
        aspect=p_aspect,
        errorbar=None,
        color="blue",
    )
    # Decorating plot
    x_label = (
        "Number of carbon atoms" if margin == "N_Carbon" else "Number of double bonds"
    )
    p.set_axis_labels(x_label, "Proportion", size=12)
    p.set_titles("{col_name}", size=12)
    p.tick_params(axis="x", labelsize=margin_fontsize)
    p.set(ylim=(0, 1))
    return p


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="fach.py", description="Generate fatty acid composition heatmaps"
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="i",
        required=True,
        type=pathlib.Path,
        help="path to the input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="o",
        required=True,
        type=pathlib.Path,
        help="path to the desired output directory",
    )
    parser.add_argument(
        "-s",
        "--subdir",
        dest="s",
        action="store_true",
        help="if set, saves plots to subdirectories by lipid class",
    )
    parser.add_argument(
        "-m",
        "--mean",
        dest="m",
        action="store_true",
        help="if set, adds dashed lines to the plots denoting mean values",
    )
    parser.add_argument(
        "-t",
        "--tables",
        dest="t",
        action="store_true",
        help="if set, saves intermediate data tables as .csv files",
    )
    parser.add_argument(
        "-b",
        "--bar",
        dest="b",
        action="store_true",
        help="if set, saves marginal barplots for each lipid class",
    )
    parser.add_argument(
        "-c",
        "--cmap",
        dest="c",
        required=False,
        default="copper",
        help="the desired colormapping to use for the heatmap",
    )
    parser.add_argument(
        "-p",
        "--pad",
        dest="p",
        action="store_true",
        help="if set, pads the heatmap with N carbon/DB values between min/max",
    )
    args = parser.parse_args()

    # Import data table
    area_df = pd.read_excel(args.i, header=0, index_col=0)
    area_df.index.name = "Lipid_Annotation"
    area_df.columns.name = "Sample_ID"
    # Parse lipid annotations and drop lipid features that cannot be parsed
    annotations = area_df.index.to_series().apply(parse_lipid_annotation)
    if pd.isnull(annotations).sum() > 0:
        print(
            f"{sum(pd.isnull(annotations))} lipid annotations could not be "
            "parsed and have been removed"
        )
    area_df = area_df.loc[~pd.isnull(annotations)]
    annotations.dropna(inplace=True)
    # Un-pivot the table to long format
    area_df = area_df.stack().reset_index(level=1).rename(columns={0: "Area"})
    # Parse sample IDs into sample groups
    area_df["Sample_Group"] = [
        "".join(i.split("_")[:-1]) for i in area_df["Sample_ID"].values
    ]
    # Merge parsed lipid annotations with area table
    anno_df = pd.DataFrame.from_records(
        annotations,
        index=annotations.index,
        columns=["Lipid_Class", "N_Carbon", "N_DB"],
    )
    area_df = area_df.join(anno_df)
    # Fix dtypes
    area_df = area_df.astype(
        {
            "Sample_ID": "string",
            "Area": "float",
            "Sample_Group": "category",
            "Lipid_Class": "category",
            "N_Carbon": "int32",
            "N_DB": "int32",
        }
    )

    # Create output directory
    pathlib.Path(args.o).mkdir(parents=True, exist_ok=True)
    if args.t:
        area_df.to_csv(pathlib.Path(args.o, "Parsed_Area_Table.csv", index=False))
    # Extract unique lipid classes and sample groups
    lipid_classes = area_df["Lipid_Class"].drop_duplicates().values
    sample_groups = area_df["Sample_Group"].drop_duplicates().values

    # Begin looping through the lipid classes
    for c in tqdm(lipid_classes):
        # Create an output subdirectory if the flag is set
        if args.s:
            pathlib.Path(args.o, c).mkdir(parents=True, exist_ok=True)

        # Compute average areas across replicates of the same sample group
        c_area_df = (
            area_df.loc[area_df["Lipid_Class"] == c]
            .groupby(["Lipid_Annotation", "Sample_Group"], as_index=False)
            .mean("Area")
            .rename(columns={"Area": "Average_Area"})
        )
        # Compute proportions within each Sample_Group
        c_area_df["Proportion"] = c_area_df["Average_Area"] / (
            c_area_df.groupby(["Sample_Group"])["Average_Area"].transform("sum")
        )
        if args.t:
            c_area_df.to_csv(
                pathlib.Path(args.o, c, f"{c}_Average_Area_Table.csv"), index=False
            )

        if args.b:
            p = plot_marginal_barplot(c_area_df, "N_Carbon", args.p)
            p.savefig(
                fname=pathlib.Path(args.o, c, f"{c}_N_Carbon_Marginal.png"),
                bbox_inches="tight",
            )
            plt.close()
            p = plot_marginal_barplot(c_area_df, "N_DB", args.p)
            p.savefig(
                fname=pathlib.Path(args.o, c, f"{c}_N_DB_Marginal.png"),
                bbox_inches="tight",
            )
            plt.close()

        for g in sample_groups:
            g_area_df = c_area_df.loc[
                c_area_df["Sample_Group"] == g, c_area_df.columns != "Sample_Group"
            ]
            # If only one sum composition, skip to the next lipid
            is_singleton_composition = (
                g_area_df[["N_Carbon", "N_DB"]].drop_duplicates().shape[0] == 1
            )
            if is_singleton_composition:
                continue
            # Sum together feature area values if they share N_Carbon and N_DB
            g_area_df = g_area_df.groupby(["N_Carbon", "N_DB"], as_index=False).sum()
            # Determine min and max values
            n_carbon_range = np.arange(
                g_area_df["N_Carbon"].min(), g_area_df["N_Carbon"].max() + 1, 1
            )
            n_db_range = np.arange(
                g_area_df["N_DB"].min(), g_area_df["N_DB"].max() + 1, 1
            )

            # If flag is set, pad N_Carbon and N_DB values so they are not
            # skipped along axes
            if args.p:
                g_area_df = (
                    pd.DataFrame({"N_Carbon": [n_carbon_range], "N_DB": [n_db_range]})
                    .explode("N_Carbon")
                    .explode("N_DB")
                    .merge(g_area_df, on=["N_Carbon", "N_DB"], how="outer")
                    .fillna(0)
                    .astype({"N_Carbon": "int32", "N_DB": "int32"})
                )

            # Generate FACH
            fig = plot_fach(g_area_df, args.m, args.c)
            # Save and close before moving on
            if args.s:
                fig.savefig(
                    fname=pathlib.Path(args.o, c, f"{c}_{g}.png"), bbox_inches="tight"
                )
            else:
                fig.savefig(
                    fname=pathlib.Path(args.o, f"{c}_{g}.png"), bbox_inches="tight"
                )
            plt.close()

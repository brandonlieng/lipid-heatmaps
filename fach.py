"""
fach.py

Generate fatty acid composition heatmaps from a table of area values obtained by
untargeted lipidomics.
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import re
import seaborn as sns
from tqdm.auto import tqdm


def parse_lipid_annotation(l):
    """
    Parse a lipid annotation into its corresponding lipid class, number of
    carbon atoms, and number of double bonds.

    :param l:   lipid annotation to be parsed

    :return:    list containing [lipid class, n_c, n_db]
    """
    pattern = r"^[0-z]+ *\(*([OP]{1}-)*[dt]*\d+:\d+\)*"
    # Check that the annotation has a valid format, otherwise move on to next lipid
    if not re.match(pattern, l):
        return [None, None, None]
    base_class = re.findall(r"^[0-z_]+", l)
    assert len(base_class) == 1
    base_class = base_class[0]
    composition = re.findall(r"\d+:\d+", l)[0]
    # Is this a sphingolipid/ceramide annotated MS-DIAL style?
    if re.match(r";[23]{1}O", l):
        lipid_class = base_class + "_" + re.findall(r";[23]{1}O", l)[0]
    # Is this a sphingolipid/ceramide annotated LIPIDMAPS style?
    elif re.match(r"\([dt]{1}", l):
        lipid_class = base_class + "_" + re.findall(r"\([dt]{1}", l)[0][1]
    # Is this ether- or vinyl-linked?
    elif "P-" in l or "O-" in l:
        lipid_class = base_class + "_" + re.findall(r"[PO]{1}-", l)[0][0]
    else:
        lipid_class = base_class
    [n_c, n_db] = composition.split(":")
    return [lipid_class, n_c, n_db]


def plot_fach(area_df, heatmap_cmap):
    """
    Plot a fatty acid composition heatmap.

    :param area_df:         a pandas DataFrame holding N_Carbon/N_DB data for
                            features within a lipid class and sample group
    :param heatmap_cmap:
    :param bounds:

    :return:                a list holding a Matplotlib figure and axes
    """
    # Compute mean proportional contribution values and pivot for sns.heatmap
    # tmp_df = (
    #     area_df
    #     .groupby(["N_Carbon", "N_DB"], as_index=False)
    #     ["Proportional_Contribution"]
    #     .mean()
    # )
    pad_df = []
    for n_c in range(area_df["N_Carbon"].min(), area_df["N_Carbon"].max() + 1):
        for n_db in range(area_df["N_DB"].min(), area_df["N_DB"].max() + 1):
            if (
                n_c not in area_df["N_Carbon"].values
                or n_db not in area_df["N_DB"].values
            ):
                for s in area_df["Sample_ID"].drop_duplicates():
                    pad_df.append([s, n_c, n_db, 0])
    pad_df = pd.DataFrame(
        pad_df, columns=["Sample_ID", "N_Carbon", "N_DB", "Proportional_Contribution"]
    )
    if not pad_df.empty:
        pad_df = pd.concat(
            [
                pad_df,
                (
                    area_df.groupby(["Sample_ID", "N_Carbon", "N_DB"], as_index=False)[
                        "Proportional_Contribution"
                    ].mean()
                ),
            ]
        )
    else:
        pad_df = area_df
    heatmap_df = (
        pad_df.groupby(["N_Carbon", "N_DB"], as_index=False)[
            "Proportional_Contribution"
        ]
        .mean()
        .pivot(columns="N_Carbon", index="N_DB", values="Proportional_Contribution")
    )
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
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0])
    ax_hist_y = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_cbar_pos = ax_cbar.get_position()
    ax_cbar_pos.y0 = 0.8
    ax_cbar_pos.y1 = 0.825
    ax_cbar.set_position(ax_cbar_pos)
    # Add plots to the subplot axes
    sns.heatmap(
        data=heatmap_df,
        ax=ax_heatmap,
        cbar_ax=ax_cbar,
        cmap=heatmap_cmap,
        cbar_kws={"orientation": "horizontal", "label": "Proportion"},
        mask=(heatmap_df == 0),
    )
    sns.barplot(
        data=(
            pad_df.groupby(["N_Carbon", "Sample_ID"], as_index=False)[
                "Proportional_Contribution"
            ].sum()
        ),
        x="N_Carbon",
        y="Proportional_Contribution",
        fill=False,
        color="k",
        errorbar=(
            args.ebar
            if args.ebar in ["ci", "pi", "se", "sd"]
            else (lambda x: (x.min(), x.max()))
        ),
        ax=ax_hist_x,
        width=0.8,
    )
    sns.barplot(
        data=(
            pad_df.groupby(["N_DB", "Sample_ID"], as_index=False)[
                "Proportional_Contribution"
            ].sum()
        ),
        y="N_DB",
        x="Proportional_Contribution",
        orient="h",
        fill=False,
        color="k",
        errorbar=(
            args.ebar
            if args.ebar in ["ci", "pi", "se", "sd"]
            else (lambda x: (x.min(), x.max()))
        ),
        ax=ax_hist_y,
        width=0.8,
    )
    return (fig, ax_heatmap, ax_cbar, ax_hist_x, ax_hist_y)


def plot_marginal_barplot(area_df, margin):
    """
    Plot a set of marginal distributions as barplots.

    :param area_df:         a pandas DataFrame holding N_Carbon/N_DB data for
                            features within a lipid class and sample group

    :param margin:          a string denoting the margin to be plotted; must be one of
                            "N_Carbon" or "N_DB"

    :return:                a Matplotlib figure
    """
    assert margin in ["N_Carbon", "N_DB"]
    # Sum proportions if they share the same N_Carbon or N_DB, depending on the
    # marginal varable to be plotted
    tmp_df = area_df.groupby([margin, "Sample_ID"], as_index=False)[
        "Proportional_Contribution"
    ].sum()
    pad_df = []
    for i in range(tmp_df[m].min(), tmp_df[m].max() + 1):
        if i not in tmp_df[m].values:
            for s in tmp_df["Sample_ID"].drop_duplicates():
                pad_df.append([i, s, 0])
    pad_df = pd.DataFrame(pad_df, columns=[m, "Sample_ID", "Proportional_Contribution"])
    if not pad_df.empty:
        tmp_df = pd.concat([tmp_df, pad_df])
    fig, ax = plt.subplots(figsize=(8, 4))
    p = sns.barplot(
        data=tmp_df,
        x=margin,
        y="Proportional_Contribution",
        fill=False,
        color="k",
        errorbar=(
            args.ebar
            if args.ebar in ["ci", "pi", "se", "sd"]
            else (lambda x: (x.min(), x.max()))
        ),
    )
    return (fig, ax)


if __name__ == "__main__":
    # Parse command-line arguments
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
        help="path to the output directory",
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
        help="if set, saves the parsed area table as a CSV file",
    )
    parser.add_argument(
        "-b",
        "--bar",
        dest="b",
        action="store_true",
        help="if set, saves marginal barplots for each lipid class",
    )
    parser.add_argument(
        "-a",
        "--annotate",
        dest="a",
        action="store_true",
        help="if set, annotates FACHs with average N_Carbon and N_DB values",
    )
    parser.add_argument(
        "-c",
        "--cmap",
        dest="c",
        required=False,
        default="magma_r",
        help="the colormap to use for the heatmap",
    )
    parser.add_argument(
        "-f",
        "--font",
        dest="f",
        required=False,
        default="Arial",
        help="the font to use when plotting",
    )
    parser.add_argument(
        "-l",
        "--labelsize",
        dest="l",
        required=False,
        default=10,
        type=int,
        help="the font size to use for plot labels",
    )
    parser.add_argument(
        "--carbonmin",
        dest="cmin",
        required=False,
        default=None,
        type=int,
        help="the lower N_Carbon bound to use when plotting",
    )
    parser.add_argument(
        "--carbonmax",
        dest="cmax",
        required=False,
        default=None,
        type=int,
        help="the upper N_Carbon bound to use when plotting",
    )
    parser.add_argument(
        "--dbmin",
        dest="dbmin",
        required=False,
        default=None,
        type=int,
        help="the lower N_DB bound to use when plotting",
    )
    parser.add_argument(
        "--dbmax",
        dest="dbmax",
        required=False,
        default=None,
        type=int,
        help="the upper N_DB bound to use when plotting",
    )
    parser.add_argument(
        "--ebar",
        dest="ebar",
        required=False,
        default="sd",
        choices=["ci", "pi", "se", "sd", "minmax"],
        type=str,
        help="the metric to use when plotting errorbars",
    )
    args = parser.parse_args()
    # Create output directory
    pathlib.Path(args.o).mkdir(exist_ok=True)
    # Set global plotting params
    plt.rcParams["font.family"] = args.f
    matplotlib.use("Agg")
    # Import the data matrix
    area_df = pd.read_excel(args.i, header=0, index_col=0)
    # Un-pivot the matrix to a long-format table
    area_df = (
        area_df.melt(ignore_index=False)
        .reset_index()
        .rename(columns={"variable": "Sample_ID", "value": "Area"})
    )
    # Parse lipid annotations to [Lipid_Class, N_Carbon, N_DB] and drop lipid features
    # that cannot be parsed
    anno_df = pd.DataFrame.from_records(
        area_df["Lipid_Annotation"].apply(parse_lipid_annotation),
        columns=["Lipid_Class", "N_Carbon", "N_DB"],
    )
    area_df = pd.concat([area_df, anno_df], axis=1)
    if pd.isnull(area_df["Lipid_Class"]).sum() > 0:
        unparsable = (
            area_df.loc[pd.isnull(area_df["Lipid_Class"]), "Lipid_Annotation"]
            .drop_duplicates().
            values
        )
        print(
            f"{unparsable.size} lipid annotations could not be parsed and have been "
            "removed"
        )
    with open(pathlib.Path(args.o, "Unparsable_Lipids.txt"), "w") as f:
        for m in unparsable:
            f.write(f"{m}\n")
    area_df = area_df.loc[~pd.isnull(area_df["Lipid_Class"])]
    area_df = area_df.drop(columns="Lipid_Annotation")
    # Fix dtypes
    area_df = area_df.astype(
        {
            "Sample_ID": "string",
            "Lipid_Class": "category",
            "N_Carbon": "int32",
            "N_DB": "int32",
            "Area": "float",
        }
    )
    # Sum lipid feature areas within samples if they share the same sum composition
    area_df = area_df.groupby(
        ["Sample_ID", "Lipid_Class", "N_Carbon", "N_DB"],
        as_index=False,
        sort=False,
        observed=True,
    ).sum()
    # Get the total area detected in each sample per lipid class
    total_sample_class_areas = (
        area_df.groupby(["Sample_ID", "Lipid_Class"], observed=True, as_index=False)["Area"]
        .sum()
        .rename(columns={"Area": "Sample_Class_Total_Area"})
    )
    # Calculate the proportion of signal (area) that each lipid species contributes to
    # the total area of it's respective lipid class (calculated per-sample)
    area_df = area_df.merge(total_sample_class_areas)
    area_df["Proportional_Contribution"] = (
        area_df["Area"] / area_df["Sample_Class_Total_Area"]
    )
    # Parse sample IDs into sample groups
    area_df.insert(
        1,
        "Sample_Group",
        ["_".join(i.split("_")[:-1]) for i in area_df["Sample_ID"].values],
    )
    # Save the area table with proportional contributions to a CSV file
    if args.t:
        area_df.to_csv(pathlib.Path(args.o, "Parsed_Area_Table.csv"), index=False)
        average_values = []
    # Begin looping through the lipid classes
    for c in tqdm(area_df["Lipid_Class"].drop_duplicates().values):
        for g in area_df["Sample_Group"].drop_duplicates().values:
            # Subset for rows relevant to this lipid class/sample group
            g_area_df = area_df.loc[
                (area_df["Lipid_Class"] == c) & (area_df["Sample_Group"] == g)
            ]
            # Drop rows for species not detected in this sample
            g_area_df = g_area_df.groupby(["N_Carbon", "N_DB"]).filter(
                lambda x: not all(x["Area"] == 0)
            )
            g_area_df.fillna({"Proportional_Contribution": 0}, inplace=True)
            # If one or no sum composition, skip to the next lipid
            is_skippable = (
                g_area_df[["N_Carbon", "N_DB"]].drop_duplicates().shape[0] <= 1
            )
            if is_skippable:
                continue
            # Create an output subdirectory for the current class
            pathlib.Path(args.o, c).mkdir(parents=True, exist_ok=True)
            # If the -b flag is set, produce marginal bar plots
            if args.b:
                for m in ["N_Carbon", "N_DB"]:
                    fig, ax = plot_marginal_barplot(g_area_df, m)
                    # Decorating plot
                    x_label = (
                        "Number of carbon atoms"
                        if m == "N_Carbon"
                        else "Number of double bonds"
                    )
                    ax.set_xlabel(x_label, size=args.l)
                    ax.set_ylabel("Proportion", size=args.l)
                    ax.set_title(f"{c} {g}", size=args.l)
                    ax.tick_params(axis="both", labelsize=args.l)
                    # Hide every other tick label if more than 15 values
                    if g_area_df[m].max() - g_area_df[m].min() >= 15:
                        if g_area_df[m].min() % 2 == 0:
                            for l in ax.xaxis.get_ticklabels()[1::2]:
                                l.set_visible(False)
                        else:
                            for l in ax.xaxis.get_ticklabels()[::2]:
                                l.set_visible(False)
                    plt.savefig(
                        fname=pathlib.Path(args.o, c, f"{c}_{g}_{m}_Marginal.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

            # Generate FACH
            fig, ax_heatmap, ax_cbar, ax_hist_x, ax_hist_y = plot_fach(
                g_area_df, args.c
            )
            # Determine means and mark them on the heatmap if the flag is set
            if args.m:
                n_carbon_range = np.arange(
                    g_area_df["N_Carbon"].min(), g_area_df["N_Carbon"].max() + 1, 1
                )
                n_db_range = np.arange(
                    g_area_df["N_DB"].min(), g_area_df["N_DB"].max() + 1, 1
                )
                n_carbon_values = g_area_df["N_Carbon"].drop_duplicates().values
                n_db_values = g_area_df["N_DB"].drop_duplicates().values
                avg_n_carbon = sum(
                    (
                        g_area_df[["N_Carbon", "Area"]].groupby("N_Carbon").sum()
                        / g_area_df["Area"].sum()
                    ).values.flatten()
                    * np.sort(n_carbon_values)
                )
                avg_n_db = sum(
                    (
                        g_area_df[["N_DB", "Area"]].groupby("N_DB").sum()
                        / g_area_df["Area"].sum()
                    ).values.flatten()
                    * np.sort(n_db_values)
                )
                if args.t:
                    average_values.append([c, g, avg_n_carbon, avg_n_db])
                if n_carbon_values.size > 1:
                    interpolated_n_carbon = np.interp(
                        avg_n_carbon, n_carbon_range, range(len(n_carbon_range))
                    )
                    ax_heatmap.axvline(
                        x=interpolated_n_carbon + 0.5,
                        linestyle="--",
                        linewidth=1,
                    )
                if n_db_values.size > 1:
                    interpolated_avg_n_db = np.interp(
                        avg_n_db, n_db_range, range(len(n_db_range))
                    )
                    ax_heatmap.axhline(
                        y=interpolated_avg_n_db + 0.5,
                        linestyle="--",
                        linewidth=1,
                    )
            if args.a:
                ax_heatmap.text(
                    x=0.5,
                    y=-0.15,
                    s=f"Avg. number of carbon atoms: {avg_n_carbon:.2f}"
                    + "      "
                    + f"Avg. number of double bonds: {avg_n_db:.2f}",
                    wrap=True,
                    transform=ax_heatmap.transAxes,
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=args.l,
                )
            # Decorating heatmap
            ax_heatmap.set_xlabel("Number of carbon atoms", size=args.l)
            ax_heatmap.set_ylabel("Number of double bonds", size=args.l)
            ax_heatmap.tick_params(labelsize=args.l)
            # Decorating top marginal barplot
            ax_hist_x.spines[["right", "top"]].set_visible(False)
            ax_hist_x.set_xlabel(None)
            ax_hist_x.set_ylabel("Proportion", size=args.l)
            ax_hist_x.tick_params(labelbottom=False, bottom=False)
            ax_hist_x.set_yticks(
                ax_hist_x.get_yticks(),
                labels=ax_hist_x.get_yticklabels(),
                fontsize=args.l,
            )
            ax_hist_x.set_ylim(bottom=0)
            # Decorating right marginal barplot
            ax_hist_y.spines[["right", "top"]].set_visible(False)
            ax_hist_y.set_xlabel("Proportion", size=args.l)
            ax_hist_y.set_ylabel(None)
            ax_hist_y.tick_params(labelleft=False, left=False)
            ax_hist_y.set_xticks(
                ax_hist_y.get_xticks(),
                labels=ax_hist_y.get_xticklabels(),
                fontsize=args.l,
            )
            ax_hist_y.set_xlim(left=0)
            # Decorating colourbar
            ax_cbar.xaxis.set_ticks_position("top")
            ax_cbar.set_xlabel("Proportion", size=args.l)
            # Save and close before moving on
            plt.savefig(
                fname=pathlib.Path(args.o, c, f"{c}_{g}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    if args.t:
        (
            pd.DataFrame(
                average_values,
                columns=["Lipid_Class", "Sample_Group", "Mean_N_Carbon", "Mean_N_DB"],
            ).to_csv(pathlib.Path(args.o, "Marginal_Means.csv"), index=False)
        )

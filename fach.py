import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
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



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="fach.py", description="Generate fatty acid composition heatmaps"
    )
    parser.add_argument(
        "-i", "--input", dest="i", required=True, type=pathlib.Path,
        help="path to the input file"
    )
    parser.add_argument(
        "-o", "--output", dest="o", required=True, type=pathlib.Path,
        help="path to the desired output directory"
    )
    parser.add_argument(
        "-m", "--mean", dest="m", action="store_true",
        help="if set, adds dashed lines to the plots denoting mean values"
    )
    parser.add_argument(
        "-c", "--cmap", dest="c", required=False, default="YlOrBr",
        help="the desired colormapping"
    )
    args = parser.parse_args()

    # Import data table
    area_df = pd.read_excel(args.i, header=0, index_col=0)
    area_df.index.name = "Lipid_Annotation"
    area_df.columns.name = "Sample_ID"

    # Parse lipid annotations and drop lipid features that cannot be parsed
    annotations = area_df.index.to_series().apply(parse_lipid_annotation)
    print(
        f"{sum(pd.isnull(annotations))} lipid annotations could not be parsed "
        "and have been removed"
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
        annotations, index=annotations.index,
        columns=["Lipid_Class", "N_Carbon", "N_DB"]
    )
    area_df = area_df.join(anno_df)

    # Fix dtypes
    area_df = area_df.astype(
        {
            "Sample_ID": "string", "Area": "float", "Sample_Group": "category",
            "Lipid_Class": "category", "N_Carbon": "int32", "N_DB": "int32"
        }
    )

    # Create the output directory
    pathlib.Path(args.o).mkdir(parents=True, exist_ok=True)

    # Extract unique lipid classes and sample groups
    lipid_classes = area_df["Lipid_Class"].drop_duplicates().values
    sample_groups = area_df["Sample_Group"].drop_duplicates().values

    # Generate a fatty acid composition heatmap for each lipid class
    for c in tqdm(lipid_classes):
        # Compute average areas based on sample groups
        avg_area_df = (
            area_df.loc[area_df["Lipid_Class"] == c]
            .groupby(["Lipid_Annotation", "Sample_Group"], as_index=False)
            .mean("Area")
        )
        for g in sample_groups:
            g_avg_area_df = avg_area_df.loc[avg_area_df["Sample_Group"] == g]

            # If only one sum composition, skip to the next lipid
            n_compositions = (
                g_avg_area_df[["N_Carbon", "N_DB"]]
                .drop_duplicates()
                .shape[0] == 1
            )
            if n_compositions <= 1:
                continue

            # Sum together feature area values if they share N_Carbon and N_DB
            g_avg_area_df = (
                g_avg_area_df.groupby(["N_Carbon", "N_DB"], as_index=False)
                .sum("Area")
            )

            # Calculate proportions
            g_avg_area_df["Proportion"] = (
                g_avg_area_df["Area"] /  g_avg_area_df["Area"].sum()
            )

            # Get DataFrames to plot marginal barplots
            n_carbon_range = np.arange(
                g_avg_area_df["N_Carbon"].min(),
                g_avg_area_df["N_Carbon"].max() + 1,
                1
            )
            n_db_range = np.arange(
                g_avg_area_df["N_DB"].min(), g_avg_area_df["N_DB"].max() + 1, 1
            )

            # Pad N_Carbon and N_DB values so they are not skipped along axes
            g_avg_area_padded_df = (
                pd.DataFrame(
                    {"N_Carbon": [n_carbon_range], "N_DB": [n_db_range]}
                )
                .explode("N_Carbon")
                .explode("N_DB")
                .merge(g_avg_area_df, on=["N_Carbon", "N_DB"], how="outer")
                .astype({"N_Carbon": "int32", "N_DB": "int32"})
            )

            # Create wide matrix for sns.heatmap
            mat = (
                g_avg_area_padded_df
                .pivot(columns="N_Carbon", index="N_DB", values="Proportion")
            )
            mat.fillna(0, inplace=True)

            # Initalize a grid of subplots
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(
                2, 2, width_ratios=(4, 1), height_ratios=(1, 4), left=0.1,
                right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1
            )
            ax_heatmap = fig.add_subplot(gs[1, 0], )
            ax_hist_x = fig.add_subplot(gs[0, 0])
            ax_hist_y = fig.add_subplot(gs[1, 1])
            ax_cbar = fig.add_subplot(gs[0, 1])

            # Add plots to the subplot axes
            sns.heatmap(
                data=mat, ax=ax_heatmap, cbar_ax=ax_cbar, cmap=args.c,
                cbar_kws={"orientation": "horizontal", "label": "Proportion"},
            )
            sns.barplot(
                data=(
                    g_avg_area_padded_df
                    .groupby("N_Carbon", as_index=False)
                    .sum("Proportion")
                ), x="N_Carbon", y="Proportion", color="grey", errorbar=None,
                ax=ax_hist_x, width=0.9
            )
            sns.barplot(
                data=(
                    g_avg_area_padded_df
                    .groupby("N_DB", as_index=False)
                    .sum("Proportion")
                ), y="N_DB", x="Proportion",
                orient="h", color="grey", errorbar=None, ax=ax_hist_y,
                width=0.9
            )

            # Determine means and mark them on the heatmap if the flag is set
            if args.m:
                avg_n_carbon = sum(
                    g_avg_area_df["N_Carbon"] * g_avg_area_df["Proportion"]
                )
                avg_n_db = sum(
                    g_avg_area_df["N_DB"] * g_avg_area_df["Proportion"]
                )
                ax_heatmap.axvline(
                    x=np.interp(
                        avg_n_carbon, n_carbon_range, range(len(n_carbon_range))
                    ) + 0.5, linestyle="--", linewidth=1
                )
                ax_heatmap.axhline(
                    y=np.interp(
                        avg_n_db, n_db_range, range(len(n_db_range))
                    ) + 0.5, linestyle="--", linewidth=1
                )

            # Decorating plots
            ax_hist_x.spines[["right", "top"]].set_visible(False)
            ax_hist_y.spines[["right", "top"]].set_visible(False)
            ax_hist_x.tick_params(labelbottom=False, bottom=False)
            ax_hist_y.tick_params(labelleft=False, left=False)
            ax_hist_x.set_xlabel(None)
            ax_hist_y.set_ylabel(None)
            ax_heatmap.set_xlabel("Number of carbon atoms")
            ax_heatmap.set_ylabel("Number of double bonds")
            ax_cbar.xaxis.set_ticks_position("top")

            # Save and close before moving on
            fig.savefig(fname=pathlib.Path(args.o, f"{c}_{g}.png"))
            plt.close()


# lipid-heatmaps

This is a script to generate fatty acid composition heatmaps from lipidomics data.

## Prerequisites

### Input data matrix

This script accepts a CSV-formatted lipid abundance matrix, where rows are lipid features and columns are samples. The first row of the sheet (header row) should hold a cell containing `Lipid_Annotation` followed by sample IDs. The first column of the sheet should contain the `Lipid_Annotation` header followed by lipid feature IDs. The values that fill the matrix should be the peak areas of each lipid found in each sample (`NA` values are treated as 0s).

### Sample IDs

Sample/column names should be in the format `XYZ_123`, where everything before the last underscore (`XYZ`) in each column name will be treated as a group ID and everything after (`123`) will be treated as a replicate ID. Group IDs can have underscores in them, so something like: `[DMSO_COLD_1, DMSO_COLD_2, DMSO_HOT_1, DMSO_HOT_2]` is OK. Group IDs will be used to name output files, so try and pick sensible names -- choose something like "AML_T_Pool_1" over "AML-Drug treatment (pooled)_1".

### Lipid feature IDs

This script parses:
* Glycerophospholipids as sum compositions (e.g., `PC 34:1`, `PC(34:1)`, or `PC (34:1)`)
* Ether- and vinyl-linked glycerophospholipids as sum compositions (e.g., `LPE(O-18:2/0:0)`)
* Ceramides (including dihydroceramides, phytoceramides, and hydroxyceramides) as backbone/fatty acid resolved species (e.g., `Cer(d18:1/16:0)`, `Cer(d18:0/16:0)`, `Cer(t18:1/16:0)`, or `Cer(d18:1/16:0(OH))`)

The script will automatically parse these into their respective lipid class, number of carbons, and number of double bonds. Unparsable lipid annotations will be removed from the matrix prior to generating figures. The number of dropped features is printed to the terminal when the script is run and these features are written to `Unparsable_Lipids.txt`.

### Installing dependencies

You'll need to have Python installed on your system to use this tool. A `requirements.txt` file is provided to install required dependencies with `python -m pip install -r requirements.txt`.

## Usage

Use a Terminal (Mac) or CMD session (Windows) to run:

```
python3 fach.py [-h] -i I -o O [-m] [-t] [-b] [-a] [-c C] [-f F] [-l L] [-g] [-e {ci,pi,se,sd,mixmax}]
```

where:

```
  -h, --help            show a help message and exit
  -i, --input I         path to the input file
  -o, --output O        path to the output directory
  -m, --mean            if set, adds dashed lines to the plots denoting mean values
  -t, --tables          if set, saves the parsed area and marginal means tables as CSV files
  -b, --bar             if set, saves marginal barplots for each lipid class
  -a, --annotate        if set, annotates FACHs with average N_Carbon and N_DB values
  -c, --cmap C          the colormap to use for the heatmap
  -f, --font F          the font to use when plotting
  -l, --labelsize L     the font size to use for plot labels
  -g, --groupaxes       if set, uses the same axes range for all FACHs within a sample group
  -e, --ebar {ci,pi,se,sd,minmax}
                        the metric to use when plotting errorbars
```

Arguments in square brackets are optional.

Example: `python fach.py -i ~/Documents/example/area_table.csv -o ~/Desktop/output_fach_plots -m -c copper -b` will generate fatty acid composition heatmaps using areas stored in input file `~/Documents/example/area_table.csv`. These plots will be saved as png files in `~/Desktop/output_fach_plots`. Dashed lines will appear denoting mean values and the colour map used will be `copper`. Barplots showing marginal distributions across groups will be plotted. Axis ranges will not be determined on a per-group basis, so two FACH plots for the same lipid class but for different sample groups may have different axis ranges.

Notes:
* To make the file paths easy, you can copy a file from the Finder/File Explorer and paste it to the Terminal/CMD as a file path
* Colour palettes that can be passed via `-c` can be found in the [Matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html) and [seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html) documentation

# lipid-heatmaps

This is a script to generate fatty acid composition heatmaps from lipidomics data.

## Prerequisites

### Formatting your input data

This script accepts an `.xlsx` Excel file where rows are features and columns are samples. The first row of the sheet should hold a blank cell followed by sample IDs. The first column of the sheet should hold a blank cell followed by feature annotations. The values that fill the matrix should be the peak areas of each lipid found in each sample (`NA` values are OK, they will be treated as 0s).

Sample/column names should be in the format `XYZ_123`. Everything before (`XYZ`) the last underscore in each column name will be treated as a group ID and everything after (`123`) will be treated as a replicate ID. Group IDs can have underscores in them, so something like: `[KEGG_ID, DMSO_COLD_1, DMSO_COLD_2, DMSO_HOT_1, DMSO_HOT_2]` is OK. Group IDs will be used to name output files, so try and pick sensible names -- choose something like "AML_T_Pool_1" over "AML-Drug treatment (pooled)_1".

Lipid annotations should be in the format `ABC 12:3`. The script will automatically parse these into lipid class (`ABC`), number of carbons (`12`), and number of double bonds (`3`). Currently, the script supports simple annotations like `PC 34:4` or `CAR 40:2`, as well as ether linked classes like `PE P-38:3`. Unparsable lipid annotations will be removed from the matrix prior to generating figures. The number of dropped features is printed to the terminal when the script is run.

### Installing dependencies

You'll need to have Python installed on your system to use this tool. A `requirements.txt` file is provided to install required dependencies with `python3 -m pip install -r requirements.txt`.

### Usage instructions

Use a Terminal (Mac) or CMD session (Windows) to run:

```
python3 fach.py [-h] -i I -o O [-s] [-m] [-t] [-b] [-c C] [-p]
```

where:

```
-h                  if set, prints the help message

-i I, --input I     path to the input file

-o O, --output O    path to the desired output directory

-s, --subdir        if set, saves plots to subdirectories by lipid class

-m, --mean          if set, adds dashed lines to the plots denoting mean values

-t, --tables        if set, saves intermediate data tables as .csv files

-b, --bar           if set, saves marginal barplots for each lipid class

-c C, --cmap C      the desired colormapping to use for the heatmap
                    (default: 'copper')

-p, --pad           if set, pads the heatmap with N carbon/DB values between
                    min/max
```

Arguments in square brackets are optional.

Example: `python3 fach.py -i ~/Documents/example/area_table.xlsx -o ~/Desktop/output_fach_plots -m` will generate fatty acid composition heatmaps using areas stored in input file `~/Documents/example/area_table.xlsx`. These plots will be saved as png files in `~/Desktop/output_fach_plots`. Dashed lines will appear denoting mean values and the colour map used will be `YlOrBl`.

Notes:
* To make the file paths easy, you can copy a file from the Finder/File Explorer and paste it to the Terminal/CMD as a file path
* Colour palettes that can be passed via `-c` can be found in the [Matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html) and [seaborn](https://seaborn.pydata.org/tutorial/color_palettes.html) documentation

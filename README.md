This repository contains the cleaned scripts for Boosted Regression Tree (BRT) modelling and bioinformatics analyses of Nairobi sheep disease virus (NSDV) risk.

The BRT workflow was converted from notebook-style analyses into standalone Python scripts. All required Python libraries can be installed using the packages listed in `requirements.txt`. The scripts read input files from the local `input/` directory and write generated results to the local `output/` directory by default.

Contents:

`brt_scripts/tick.py`: Runs BRT models for the three tick vectors and exports model metrics, predicted probabilities, high-risk counties, and relative-contribution tables.

`brt_scripts/nsdv.py`: Runs the NSDV BRT model using environmental variables and predicted probabilities from the three tick-vector models.

`input/`: Includes all required BRT input files, including risk factors, tick and NSDV occurrence records, and inverse-probability weights.

`bioinformatics_scripts/Bowtie.sh`: Uses Bowtie2 to align sequencing reads to reference sequences and exports sorted BAM files.

`bioinformatics_scripts/SRA_Bowtie_NSDV.sh`: Downloads SRA reads, converts them to FASTQ, performs read cleaning, and maps cleaned reads to the NSDV reference.

`bioinformatics_scripts/Trinity_NSDV.sh`: Cleans raw reads, removes rRNA reads, and assembles transcripts using Trinity.

`bioinformatics_scripts/BLASTN_NSDV.sh` and `bioinformatics_scripts/DIAMOND_BLASTX_NSDV.sh`: Compare assembled contigs against GenBank nucleotide and protein databases to identify homologous sequences.

Recommended running order for BRT models:

```powershell
pip install -r requirements.txt
python brt_scripts\tick.py
python brt_scripts\nsdv.py
```

Optional settings:

Set `BRT_OUTPUT_DIR` to write results to a different output directory.

Tick and NSDV high-risk GAUL2 areas are defined using `mean_prob >= 0.5`. Tick and NSDV relative-contribution tables are exported as decimal values with four digits after the decimal point.

Generated BRT results are written to `output/` by default.

If you encounter any bugs or issues with the code, please feel free to get in touch.

#!/bin/bash

files=(
Library_ID
)

READ_DIR=/path/raw_reads
OUT_DIR=/path/trinity_output
RRNA_INDEX=/path/SILVA_132_rRNA.fasta.bowtie2
THREADS=16
MAX_MEMORY=128G

for file in "${files[@]}"; do
    fastp \
        -i "$READ_DIR/${file}_1.fq.gz" \
        -I "$READ_DIR/${file}_2.fq.gz" \
        -o "$OUT_DIR/${file}.clean_1.fq" \
        -O "$OUT_DIR/${file}.clean_2.fq" \
        -f 2 -F 2 \
        -5 --cut_front_window_size 1 --cut_front_mean_quality 15 \
        -3 --cut_tail_window_size 1 --cut_tail_mean_quality 15 \
        -r --cut_right_window_size 4 --cut_right_mean_quality 15 \
        --length_required 25 \
        --correction \
        --thread "$THREADS"

    bowtie2 \
        -p "$THREADS" --no-mixed --no-discordant -k 1 \
        -x "$RRNA_INDEX" \
        -1 "$OUT_DIR/${file}.clean_1.fq" \
        -2 "$OUT_DIR/${file}.clean_2.fq" \
        --un-conc "$OUT_DIR/${file}.norRNA_%.fq" \
        --no-unal \
        -S "$OUT_DIR/${file}.rRNA.sam"

    Trinity \
        --seqType fq \
        --max_memory "$MAX_MEMORY" \
        --CPU "$THREADS" \
        --left "$OUT_DIR/${file}.norRNA_1.fq" \
        --right "$OUT_DIR/${file}.norRNA_2.fq" \
        --output "$OUT_DIR/${file}.trinity" \
        --full_cleanup
done

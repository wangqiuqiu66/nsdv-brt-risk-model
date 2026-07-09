#!/bin/bash

files=(
Library_ID
)

READ_DIR=/path/clean_reads
REF_FASTA=/path/NSDV_contigs.fasta
OUT_DIR=/path/nsdv_contig_mapping
INDEX="$OUT_DIR/nsdv_contigs.bowtie2"
THREADS=15

bowtie2-build --threads "$THREADS" "$REF_FASTA" "$INDEX"

for file in "${files[@]}"; do
    bowtie2 \
        --end-to-end --no-mixed --no-discordant \
        -p "$THREADS" \
        -x "$INDEX" \
        -1 "$READ_DIR/${file}_1.fq.gz" \
        -2 "$READ_DIR/${file}_2.fq.gz" \
        | samtools view -h -b -F4 \
        | samtools sort -@ "$THREADS" -o "$OUT_DIR/${file}.sorted.bam"
done

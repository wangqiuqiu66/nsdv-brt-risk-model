#!/bin/bash

files=(
Library_ID
)

READ_DIR=/path/clean_reads
COI_REF=/path/tick_COI_reference.fasta
OUT_DIR=/path/coi_mapping
THREADS=10

mkdir -p "$OUT_DIR"

bowtie2-build \
    --threads "$THREADS" \
    "$COI_REF" \
    "$OUT_DIR/tick_coi.bowtie2"

for file in "${files[@]}"; do
    bowtie2 \
        -p "$THREADS" --no-mixed --no-discordant --end-to-end -k 2 \
        -x "$OUT_DIR/tick_coi.bowtie2" \
        -1 "$READ_DIR/${file}_1.fq.gz" \
        -2 "$READ_DIR/${file}_2.fq.gz" \
        2> "$OUT_DIR/${file}.bowtie2.log" \
        | samtools view -h -b -F4 \
        > "$OUT_DIR/${file}.bam"

    samtools sort "$OUT_DIR/${file}.bam" > "$OUT_DIR/${file}.sorted.bam"
    samtools coverage "$OUT_DIR/${file}.sorted.bam" > "$OUT_DIR/${file}.coverage.txt"
    samtools depth -a "$OUT_DIR/${file}.sorted.bam" > "$OUT_DIR/${file}.depth.txt"

    rm "$OUT_DIR/${file}.bam"
done

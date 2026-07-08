#!/bin/bash

files=(
Library_ID
)

READ_DIR=/path/clean_reads
REF_FASTA=/path/NSDV_contigs.fasta
OUT_DIR=/path/nsdv_contig_mapping
THREADS=15

mkdir -p "$OUT_DIR"

bowtie2-build \
    --threads "$THREADS" \
    "$REF_FASTA" \
    "$OUT_DIR/nsdv_contigs.bowtie2"

for file in "${files[@]}"; do
    bowtie2 \
        --end-to-end --no-mixed --no-discordant \
        -p "$THREADS" \
        -x "$OUT_DIR/nsdv_contigs.bowtie2" \
        -1 "$READ_DIR/${file}_1.fq.gz" \
        -2 "$READ_DIR/${file}_2.fq.gz" \
        -S "$OUT_DIR/${file}.sam"

    samtools view -bSF4 "$OUT_DIR/${file}.sam" > "$OUT_DIR/${file}.bam"
    samtools sort -@ "$THREADS" "$OUT_DIR/${file}.bam" -o "$OUT_DIR/${file}.sorted.bam"
    samtools rmdup "$OUT_DIR/${file}.sorted.bam" "$OUT_DIR/${file}.sorted.rmdup.bam"
    samtools index "$OUT_DIR/${file}.sorted.rmdup.bam"
    samtools coverage "$OUT_DIR/${file}.sorted.rmdup.bam" > "$OUT_DIR/${file}.coverage.txt"
    samtools depth -a "$OUT_DIR/${file}.sorted.rmdup.bam" > "$OUT_DIR/${file}.depth.txt"

    rm "$OUT_DIR/${file}.sam" "$OUT_DIR/${file}.bam" "$OUT_DIR/${file}.sorted.bam"
done

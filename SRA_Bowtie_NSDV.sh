#!/bin/bash

PROJECT_ID=Project_ID
SRA_LIST=/path/SRA_list/${PROJECT_ID}.txt
SRA_DIR=/path/sra_files/${PROJECT_ID}
FASTQ_DIR=/path/fastq/${PROJECT_ID}
OUT_DIR=/path/SRA_Bowtie_NSDV/${PROJECT_ID}
BOWTIE2_INDEX=/path/NSDV_reference.mapping.fasta.bowtie2
THREADS=16

mkdir -p "$SRA_DIR" "$FASTQ_DIR" "$OUT_DIR"

while read -r sra_id; do
    [ -z "$sra_id" ] && continue

    prefetch "$sra_id" -O "$SRA_DIR"

    fastq-dump \
        "$SRA_DIR/${sra_id}/${sra_id}.sra" \
        --split-3 \
        --gzip \
        -O "$FASTQ_DIR"

    fastp \
        -i "$FASTQ_DIR/${sra_id}_1.fastq.gz" \
        -I "$FASTQ_DIR/${sra_id}_2.fastq.gz" \
        -o "$OUT_DIR/${sra_id}.clean_1.fq.gz" \
        -O "$OUT_DIR/${sra_id}.clean_2.fq.gz" \
        --low_complexity_filter \
        --detect_adapter_for_pe \
        --length_required 25 \
        --correction \
        --dedup \
        --dont_eval_duplication \
        --thread "$THREADS"

    bowtie2 \
        --end-to-end --no-mixed --no-discordant \
        -p "$THREADS" \
        -x "$BOWTIE2_INDEX" \
        -1 "$OUT_DIR/${sra_id}.clean_1.fq.gz" \
        -2 "$OUT_DIR/${sra_id}.clean_2.fq.gz" \
        -S "$OUT_DIR/${sra_id}.sam"

    samtools view -bSF4 "$OUT_DIR/${sra_id}.sam" > "$OUT_DIR/${sra_id}.bam"
    samtools sort -@ "$THREADS" "$OUT_DIR/${sra_id}.bam" -o "$OUT_DIR/${sra_id}.sorted.bam"
    samtools rmdup "$OUT_DIR/${sra_id}.sorted.bam" "$OUT_DIR/${sra_id}.sorted.rmdup.bam"
    samtools index "$OUT_DIR/${sra_id}.sorted.rmdup.bam"
    samtools idxstats "$OUT_DIR/${sra_id}.sorted.rmdup.bam" > "$OUT_DIR/${sra_id}_mapping.txt"

    rm "$OUT_DIR/${sra_id}.sam" "$OUT_DIR/${sra_id}.bam" "$OUT_DIR/${sra_id}.sorted.bam"
done < "$SRA_LIST"

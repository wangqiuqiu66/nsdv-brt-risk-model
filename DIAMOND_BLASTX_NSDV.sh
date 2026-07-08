#!/bin/bash

CONTIG_FASTA=contig.fasta
NR_DB=/path/GenBank_nr_database/nr.dmnd
OUT_PREFIX=file
THREADS=20

/path/diamond blastx \
    --threads "$THREADS" \
    -e 1E-5 \
    -k 3 \
    -f 6 qseqid qlen sseqid stitle pident length evalue sskingdoms staxids \
    --tmpdir /dev/shm \
    --block-size 10 \
    --index-chunks 1 \
    -d "$NR_DB" \
    -q "$CONTIG_FASTA" \
    -o "${OUT_PREFIX}.diamond.m6" \
    --more-sensitive

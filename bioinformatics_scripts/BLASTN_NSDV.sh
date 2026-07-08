#!/bin/bash

CONTIG_FASTA=contig.fasta
NT_DB=/path/GenBank_nt_database/nt
OUT_PREFIX=file
THREADS=20

/path/blastn \
    -query "$CONTIG_FASTA" \
    -db "$NT_DB" \
    -out "${OUT_PREFIX}.blastn.m6" \
    -evalue 1E-10 \
    -max_target_seqs 3 \
    -num_threads "$THREADS" \
    -outfmt "6 qseqid qlen sacc stitle pident length evalue sskingdoms staxids"

#!/usr/bin/env bash
set -euo pipefail

# Submit the streaming+memmap pipeline with PBS dependencies.
# Assumes you have a conda env called "cognito" (edit PBS files if different).

jid0=$(qsub hpc/legacy_single_target/00_extract.pbs)
echo "submitted extract: $jid0"

jid1=$(qsub -W depend=afterok:${jid0} hpc/01_make_split_stream.pbs)
echo "submitted split:   $jid1"

jid2=$(qsub -W depend=afterok:${jid1} hpc/02_build_features_memmap.pbs)
echo "submitted feats:   $jid2"

jid3=$(qsub -W depend=afterok:${jid2} hpc/10_train_m2_memmap_gpu.pbs)
echo "submitted train:   $jid3"

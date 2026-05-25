#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/tmp_sort"
mkdir -p data

pigz -dc -p 16 data/all_ensg.txt.gz | perl -ne '
  if (/\x27GeneName\x27:\s*\x27([^'\''"]+)\x27,\s*\x27miRNA\x27:\s*\x27([^'\''"]+)\x27,\s*\x27Prediction\x27:\s*([-+0-9.eE]+)/) {
    print "$1\t$2\t$3\t$_";
  }
' | LC_ALL=C sort --parallel=32 -S 200G -T "$HOME/tmp_sort" -t $'\t' -k1,1 -k2,2 -k3,3gr | awk -F'\t' '!seen[$1 FS $2]++ { $1=$2=$3=""; sub(/^\t\t\t/, ""); print }' | pigz -p 16 > data/best_per_pair.txt.gz
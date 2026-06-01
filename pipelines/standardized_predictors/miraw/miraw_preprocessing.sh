#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input="${1:-"$script_dir/data/all_ensgs.txt.gz"}"
output="${2:-"$script_dir/data/best_per_pair.tsv.gz"}"
threads="${MIRAW_THREADS:-16}"
sort_memory="${MIRAW_SORT_MEMORY:-200G}"
tmp_dir="${TMPDIR:-$HOME/tmp_sort}"

for tool in pigz perl sort awk; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required command: $tool" >&2
    exit 127
  fi
done

if [[ ! -f "$input" ]]; then
  echo "Missing miRAW input file: $input" >&2
  exit 1
fi

mkdir -p "$tmp_dir" "$(dirname "$output")"
tmp_output="${output}.tmp"
trap 'rm -f "$tmp_output"' EXIT

printf 'Ensembl_ID\tRaw_Gene_Name\tmiRNA_Name\tScore\n' | pigz -p "$threads" > "$tmp_output"

pigz -dc -p "$threads" "$input" |
  perl -ne '
    next unless /["'\'']GeneName["'\'']\s*:\s*["'\'']([^"'\'']+)["'\'']/;
    $gene = $1;
    next unless /["'\'']miRNA["'\'']\s*:\s*["'\'']([^"'\'']+)["'\'']/;
    $mirna = $1;
    next unless /["'\'']Prediction["'\'']\s*:\s*([-+0-9.eE]+)/;
    $score = $1;
    next unless $gene =~ /^([^_.]+)(?:\.[0-9]+)?__(.*)$/;
    $ensembl = $1;
    $raw_gene = $2;
    $raw_gene = "" if $raw_gene eq "" || $raw_gene =~ /^biotype=/;
    print "$ensembl\t$mirna\t$score\t$raw_gene\n";
  ' |
  LC_ALL=C sort --parallel="$threads" -S "$sort_memory" -T "$tmp_dir" \
    -t $'\t' -k1,1 -k2,2 -k3,3gr |
  awk -F '\t' 'BEGIN { OFS = FS } !seen[$1 FS $2]++ { print $1, $4, $2, $3 }' |
  pigz -p "$threads" >> "$tmp_output"

mv "$tmp_output" "$output"
trap - EXIT

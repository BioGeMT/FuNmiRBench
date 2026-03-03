"""
GEO Download Pipeline - Downloads FASTQ files from GEO/SRA

This script is called by the RNA-seq pipeline to download raw sequencing data.
It resolves GSM sample IDs to SRR run accessions and downloads FASTQ files.

Usage:
    python geo_download.py --tsv test_experiments.tsv --output data/raw/
"""

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from Bio import Entrez
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# NCBI requires an email for Entrez API
Entrez.email = "zacharopoulou.eli@gmail.com"


@dataclass
class SRRInfo:
    """Holds information about an SRA run."""
    srr: str
    gsm: str
    layout: str  # "SINGLE" or "PAIRED"
    
    def get_fastq_files(self):
        if self.layout == "PAIRED":
            return [f"{self.srr}_1.fastq.gz", f"{self.srr}_2.fastq.gz"]
        return [f"{self.srr}.fastq.gz"]


# Resolve a single GSM accession to SRR run(s)
def resolve_gsm_to_srr(gsm_id):
    logger.info(f"Resolving {gsm_id} to SRR...")
    
    handle = Entrez.esearch(db="gds", term=gsm_id)
    record = Entrez.read(handle)
    handle.close()
    
    gds_ids = record.get("IdList", [])
    if not gds_ids:
        raise ValueError(f"No GDS IDs found for {gsm_id}")
    
    handle = Entrez.elink(dbfrom="gds", db="sra", id=gds_ids)
    linkset = Entrez.read(handle)
    handle.close()
    
    if not linkset or not linkset[0].get("LinkSetDb"):
        raise ValueError(f"No SRA link found for {gsm_id}")
    
    sra_ids = [link["Id"] for link in linkset[0]["LinkSetDb"][0]["Link"]]
    
    results = []
    for sra_id in sra_ids:
        handle = Entrez.efetch(db="sra", id=sra_id, rettype="runinfo", retmode="text")
        csv_text = handle.read()
        handle.close()
        
        if isinstance(csv_text, bytes):
            csv_text = csv_text.decode("utf-8")
        
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            sample_name = row.get("SampleName", "")
            sample_id = row.get("Sample", "")
            
            if gsm_id in (sample_name, sample_id):
                srr = row.get("Run", "")
                layout = row.get("LibraryLayout", "SINGLE").upper()
                
                if srr:
                    results.append(SRRInfo(srr=srr, gsm=gsm_id, layout=layout))
                    logger.info(f"  {gsm_id} -> {srr} ({layout})")
    
    if not results:
        raise ValueError(f"No SRR runs found for {gsm_id}")
    
    return results


# Resolve multiple GSM accessions to SRR runs
def resolve_gse_samples(gsm_ids):
    all_results = []
    for gsm_id in gsm_ids:
        try:
            srr_infos = resolve_gsm_to_srr(gsm_id)
            all_results.extend(srr_infos)
        except ValueError as e:
            logger.warning(f"Failed to resolve {gsm_id}: {e}")
    return all_results


# Parse the experiments TSV file to extract sample information
def parse_experiment_metadata(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    
    experiments = []
    for _, row in df.iterrows():
        gse_url = row.get("gse_url", "")
        gse = gse_url.split("acc=")[-1] if "acc=" in gse_url else ""
        
        control_samples = str(row.get("control_samples", "")).split(",")
        condition_samples = str(row.get("condition_samples", "")).split(",")
        
        experiments.append({"gse": gse,
                            "mirna": row.get("mirna_name", ""),
                            "experiment_type": row.get("experiment_type", ""),
                            "control_samples": [s.strip() for s in control_samples if s.strip()],
                            "condition_samples": [s.strip() for s in condition_samples if s.strip()]})
    
    return experiments


# Build ENA FASTQ URLs from SRR accession
def get_ena_fastq_urls(srr, layout):
    """
    Build ENA FTP URLs for FASTQ files.
    
    ENA URL pattern:
    ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{first6}/{padding}/{srr}/{srr}_{1,2}.fastq.gz
    
    - first6: First 6 characters of SRR (e.g., SRR843)
    - padding: For SRR IDs > 9 chars, add "00X" where X is last digit
    - For SRR IDs <= 9 chars, no padding directory
    """
    base_url = "ftp://ftp.sra.ebi.ac.uk/vol1/fastq"
    first6 = srr[:6]
    
    # Determine if padding is needed (SRR IDs > 9 characters)
    if len(srr) > 9:
        # Padding is "00" + last digit(s) to make 3 chars
        padding = srr[9:].zfill(3)
        path = f"{base_url}/{first6}/{padding}/{srr}"
    else:
        path = f"{base_url}/{first6}/{srr}"
    
    if layout == "PAIRED":
        return [f"{path}/{srr}_1.fastq.gz", f"{path}/{srr}_2.fastq.gz"]
    else:
        # Single-end: ENA stores as {srr}.fastq.gz (no _1 suffix)
        return [f"{path}/{srr}.fastq.gz"]


# Download FASTQ files from ENA (fallback)
def download_from_ena(srr, output_dir, layout):
    """
    Download pre-compressed FASTQ files directly from ENA.
    Returns list of downloaded file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    urls = get_ena_fastq_urls(srr, layout)
    downloaded_files = []
    
    for url in urls:
        filename = url.split("/")[-1]
        # For SE files, ENA uses {srr}.fastq.gz which is what we want
        output_path = output_dir / filename
        
        logger.info(f"  Downloading from ENA: {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            downloaded_files.append(output_path)
            logger.info(f"  Downloaded: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            # Clean up partial downloads
            for f in downloaded_files:
                if f.exists():
                    f.unlink()
            raise RuntimeError(f"ENA download failed for {url}: {e}")
    
    return downloaded_files


# Download FASTQ files from NCBI/SRA
def download_from_ncbi(srr, output_dir, threads, layout):
    """
    Download and convert SRA files using prefetch + fasterq-dump.
    Returns list of compressed FASTQ file paths.
    """
    output_dir = Path(output_dir)
    
    # Prefetch the SRA file
    logger.info(f"  Prefetching {srr}...")
    cmd_prefetch = ["prefetch", srr, "--output-directory", str(output_dir)]
    result = subprocess.run(cmd_prefetch, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Prefetch failed for {srr}: {result.stderr}")
    
    # Convert to FASTQ
    logger.info(f"  Converting {srr} to FASTQ...")
    cmd_fasterq = ["fasterq-dump", srr,
                    "--outdir", str(output_dir),
                    "--temp", str(output_dir),
                    "--threads", str(threads),
                    "--split-files"]
    result = subprocess.run(cmd_fasterq, capture_output=True, text=True, cwd=str(output_dir))
    if result.returncode != 0:
        raise RuntimeError(f"fasterq-dump failed for {srr}: {result.stderr}")
    
    # Clean up SRA prefetch directory
    sra_dir = output_dir / srr
    if sra_dir.is_dir():
        logger.info(f"  Cleaning up SRA directory {sra_dir}...")
        shutil.rmtree(sra_dir)
    
    # Clean up fasterq-dump temp files
    for tmp_file in output_dir.glob("fasterq.tmp.*"):
        logger.info(f"  Removing temp file {tmp_file.name}...")
        if tmp_file.is_dir():
            shutil.rmtree(tmp_file)
        else:
            tmp_file.unlink()
    
    # Compress FASTQ files with pigz (parallel, removes original by default)
    logger.info(f"  Compressing FASTQ files...")
    fastq_files = list(output_dir.glob(f"{srr}*.fastq"))
    compressed_files = []
    for fq in fastq_files:
        gz_path = Path(str(fq) + ".gz")
        # Use pigz if available, otherwise gzip (both remove original by default)
        if shutil.which("pigz"):
            subprocess.run(["pigz", "-p", str(threads), str(fq)], check=True)
        else:
            subprocess.run(["gzip", str(fq)], check=True)
        compressed_files.append(gz_path)
    
    # For single-end data, rename _1.fastq.gz to .fastq.gz
    if layout == "SINGLE":
        renamed_files = []
        for gz_file in compressed_files:
            if "_1.fastq.gz" in gz_file.name:
                new_name = gz_file.parent / gz_file.name.replace("_1.fastq.gz", ".fastq.gz")
                gz_file.rename(new_name)
                renamed_files.append(new_name)
                logger.info(f"  Renamed {gz_file.name} -> {new_name.name}")
            else:
                renamed_files.append(gz_file)
        compressed_files = renamed_files
    
    return compressed_files


# Download FASTQ files for a single SRR accession (with ENA fallback)
def download_srr(srr, output_dir, threads, layout="PAIRED"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded (skip if .fastq.gz files exist)
    existing_gz = list(output_dir.glob(f"{srr}*.fastq.gz"))
    if existing_gz:
        logger.info(f"Skipping {srr} - already downloaded: {[f.name for f in existing_gz]}")
        return existing_gz
    
    logger.info(f"Downloading {srr} to {output_dir}")
    
    # Try NCBI first, fall back to ENA if it fails
    try:
        compressed_files = download_from_ncbi(srr, output_dir, threads, layout)
    except Exception as ncbi_error:
        logger.warning(f"  NCBI download failed: {ncbi_error}")
        logger.info(f"  Falling back to ENA...")
        try:
            compressed_files = download_from_ena(srr, output_dir, layout)
        except Exception as ena_error:
            raise RuntimeError(f"Both NCBI and ENA downloads failed for {srr}. NCBI: {ncbi_error}, ENA: {ena_error}")
    
    logger.info(f"  Done: {[f.name for f in compressed_files]}")
    return compressed_files


# Download all FASTQ files for an experiment
def download_experiment(gse, control_samples, condition_samples, output_dir, threads):
    output_dir = Path(output_dir) / gse
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading experiment {gse}")
    logger.info(f"  Control samples: {control_samples}")
    logger.info(f"  Condition samples: {condition_samples}")
    
    all_samples = control_samples + condition_samples
    
    # Resolve all GSMs to SRRs
    logger.info("Resolving GSM accessions to SRR runs...")
    srr_infos = resolve_gse_samples(all_samples)
    
    # Create manifest to track what was downloaded
    manifest = {"gse": gse, "samples": {}}
    
    # Download each sample
    for srr_info in srr_infos:
        try:
            gsm_dir = output_dir / srr_info.gsm
            files = download_srr(srr_info.srr, gsm_dir, threads=threads, layout=srr_info.layout)
            
            group = "control" if srr_info.gsm in control_samples else "condition"
            manifest["samples"][srr_info.gsm] = {
                "srr": srr_info.srr,
                "layout": srr_info.layout,
                "group": group,
                "files": [str(f.relative_to(output_dir)) for f in files],
            }
        except Exception as e:
            logger.error(f"Failed to download {srr_info.srr}: {e}")
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")
    
    return manifest


# Main function
def main():
    parser = argparse.ArgumentParser(description="Download FASTQ files from GEO/SRA")
    parser.add_argument("--tsv", required=True, help="Path to experiments TSV file")
    parser.add_argument("--output", "-o", default="data/raw_fastq", help="Output directory")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Threads for fasterq-dump")
    
    args = parser.parse_args()
    
    # Parse experiments from TSV
    experiments = parse_experiment_metadata(args.tsv)
    output_dir = Path(args.output)
    
    # Download each experiment
    for exp in experiments:
        try:
            download_experiment(
                gse=exp["gse"],
                control_samples=exp["control_samples"],
                condition_samples=exp["condition_samples"],
                output_dir=output_dir,
                threads=args.threads)
            
        except Exception as e:
            logger.error(f"Failed to download {exp['gse']}: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

from Bio import Entrez
import csv
from io import StringIO

Entrez.email = "your_email@example.com"  # NCBI requirement

gsm_id = "GSM6437115"

# Search for the specific GSM sample
handle = Entrez.esearch(db="gds", term=gsm_id)
record = Entrez.read(handle)
handle.close()

gds_ids = record["IdList"]
if not gds_ids:
    raise ValueError("No GDS IDs found for " + gsm_id)

# Link from GDS to SRA
handle = Entrez.elink(dbfrom="gds", db="sra", id=gds_ids)
linkset = Entrez.read(handle)
handle.close()

sra_ids = [link["Id"] for link in linkset[0]["LinkSetDb"][0]["Link"]]

# Fetch run info as CSV and parse SRR - filter by GSM ID
srrs = []
layout_info = []
for sra_id in sra_ids:
    handle = Entrez.efetch(db="sra", id=sra_id, rettype="runinfo", retmode="text")
    csv_text = handle.read()
    handle.close()
    
    # Decode bytes to string if necessary
    if isinstance(csv_text, bytes):
        csv_text = csv_text.decode('utf-8')

    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        # Only include SRRs that match our specific GSM sample
        if row.get("SampleName") == gsm_id or row.get("Sample") == gsm_id:
            srr = row["Run"]
            srrs.append(srr)
            layout = row.get("LibraryLayout", "UNKNOWN")
            
            # Generate FASTQ file names based on layout
            if layout == "PAIRED":
                fastq_files = [f"{srr}_1.fastq.gz", f"{srr}_2.fastq.gz"]
            else:  # SINGLE
                fastq_files = [f"{srr}.fastq.gz"]
            
            layout_info.append({
                "SRR": srr,
                "Layout": layout,
                "FASTQ_files": fastq_files
            })

print("SRRs for", gsm_id, ":", srrs)
print("\nLayout and FASTQ file information:")
for info in layout_info:
    print(f"  {info['SRR']}: {info['Layout']}")
    print(f"    FASTQ files: {', '.join(info['FASTQ_files'])}")

# FuNmiRBench

Benchmarking framework for **functional miRNA target prediction**.

FuNmiRBench provides:

- Standardized metadata for >50 GEO functional miRNA perturbation datasets  
- A unified Python API to:
  - list and filter experiments by miRNA, cell line, perturbation type, tissue, GEO accession  
  - load differential expression tables (edgeR outputs) into pandas  
  - summarize the available datasets
- A foundation for baseline models and future dashboards (visualization & evaluation)

---

## 🔧 Installation (development setup)

Clone the repo:

```bash
git clone git@github.com:BioGeMT/FuNmiRBench.git
cd FuNmiRBench

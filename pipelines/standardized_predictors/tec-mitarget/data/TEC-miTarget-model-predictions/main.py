import pandas as pd

for i in range(10):
    dir_path = f'test_split_{i}'
    outPathAll = dir_path + '/predict.tsv'
    outPathPos = dir_path + '/predict-positive.tsv'

    df=pd.read_csv(outPathAll,sep='\t')
    idx = df.groupby(['query_ids', 'target_ids'])['predictions'].idxmax()
    df=df.loc[idx]
    df = df[["query_ids", "target_ids", "predictions"]]
    df.to_csv(outPathAll[:-4]+'-gene-level.tsv', sep="\t", index=False)
    df=pd.read_csv(outPathPos,sep='\t')
    idx = df.groupby(['query_ids', 'target_ids'])['predictions'].idxmax()
    df=df.loc[idx]
    df = df[["query_ids", "target_ids", "predictions"]]
    df.to_csv(outPathAll[:-4]+'-gene-level.tsv', sep="\t", index=False)
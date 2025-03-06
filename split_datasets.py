from helpers import CustomTrial, CustomDataLoader, load_data
import pandas as pd

datasets = ['davis', 'kiba']
for dataset in datasets:
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)
    folds = [0, 1, 2, 3, 4]
    for fold, idx_val in enumerate(val_folds):
        if fold not in folds:
            continue
        df_train = df_train_val[~ df_train_val.index.isin(idx_val)]
        df_val = df_train_val[df_train_val.index.isin(idx_val)]
    df_train.to_csv(f"./{dataset}_train.csv", index=False)
    df_test.to_csv(f"./{dataset}_test.csv", index=False)
    df_val.to_csv(f"./{dataset}_val.csv", index=False)
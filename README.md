# miaomiao
## Create your conda environment
You can create a conda environment by referencing the environment.yml file.
## Split your datasets
Davis and KIBA datasets were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data. Alternatively, you can use the datasets from the davis.zip and kiba.zip compressed files. We refer to https://github.com/vtarasv/3d-prot-dta, if you need to split the dataset using the same method as ours, you can download it. Then transfer our split_datasets.py to the corresponding folder and execute python split_datasets.py.
## Create your datasets
Base learners were downloaded from:
1.https://github.com/thinng/GraphDTA
2.https://github.com/lizongquan01/TEFDTA
3.https://github.com/syc2017/KCDTA
4.https://github.com/vtarasv/3d-prot-dta
5.https://github.com/HySonLab/Ligand_Generation
You need to run the base learners and build datasets(eg.train.csv, test.csv) based on their learning results. The detailed steps can be viewed in creat_data.py.
The CSV file must contain the following content:compound_iso_smiles, target_sequence, affinity, nox_target_sequence, dprotdta, graphdta, kcdta, liggendta, tefdta.
Then, you can run
```
python create_data.py
```
If you don't have time to create your datasets, you can download the datasets we have prepared from the link below and place these files there/ In data/processed/.
```
https://zenodo.org/records/15110569
```
## Training
You can run
```
python training.py
```
You can change the parameters by modifying config.yaml. 
## If you have any questions, please contact me and I will answer them as soon as possible.

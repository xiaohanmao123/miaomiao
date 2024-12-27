# miaomiao
## Create your datasets
Davis and KIBA datasets were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data
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
## Training
You can run
```
python training.py
```

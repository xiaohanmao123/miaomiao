# miaomiao
mldta is a deep learning framework for drug–target affinity (DTA) prediction based on dynamic fusion and auxiliary predictive modules (APMs).  
The model is evaluated on the Davis and KIBA benchmark datasets.
## Notes
This repository contains the implementation of the MLDTA model.
The corresponding manuscript is currently under review.
## Project Structure
- `create_data_train_val.py`: dataset preprocessing
- `training.py`: model training
- `predict.py`: prediction and evaluation
- `data/`: raw and processed datasets
## Create your conda environment
You can create a conda environment by referencing the requirements.txt file.
```
conda create -n dta python=3.10
conda activate dta
pip install -r requirements.txt
```
## Create your datasets
Due to GitHub's file size limitations, we have uploaded the datasets to Hugging Face:  
```
 https://huggingface.co/datasets/xiaohan123456/davis_kiba/tree/main
```
Please download the files from this link and place them in the `data/` directory before running the preprocessing scripts.The files data/davis_test.csv, davis_train.csv, davis_val.csv, kiba_test.csv, kiba_train.csv, and kiba_val.csv are pre-split datasets containing APMs prediction data. You can generate these datasets by running the following code. 
```
python create_data_train_val.py
```
After successful execution, the data/processed directory will contain the corresponding processed files: davis_test.pt, davis_train.pt, davis_val.pt, kiba_test.pt, kiba_train.pt, and kiba_val.pt.
## Training
You can run
```
python training.py 0 0 0
```
Where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the models, 0 for our model; and the third argument is for the index of the cuda, 0 for 'cuda:0'.
You can change the parameters by modifying config.yaml. 
## Predicting
You can run
```
python predict.py 0 0 0
```
Where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively; the second argument is for the index of the models, 0 for our model; and the third argument is for the index of the cuda, 0 for 'cuda:0'. After running predict.py, the script will generate either davis_test_pred.csv or kiba_test_pred.csv, which contain the model's predicted values.
## Using Custom Datasets
We provide a dummy dataset: data/dummy_test.csv, data/dummy_train.csv, data/dummy_val.csv to demonstrate that MLDTA can be adapted to user-defined drug–target affinity datasets. You can generate the processed PyTorch files by running:
```
python create_dummy_data.py
```
This will create the following files in `data/processed/`: `dummy_test.pt`, `dummy_train.pt`, `dummy_val.pt`.  
After that, you can train the model on the dummy dataset with:
```
python training_dummy.py 0 0 0
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.
## Citation
If you use this code or data, please cite our work. See `CITATION.cff` for details.

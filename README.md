Vocabulary: This folder contains character to index and index to character conversion dictionaries.

Script: This folder contains scripts used to train the model and evaluate the model performance under multiple cases.

Database access:

[1] Simulated spectroscopy database: This is the raw spectra database, refer to corresponding spectra by database['spec_category']['molecule_smiles'] eg. database['ms']['c1ccccc1'] 

https://figshare.com/articles/dataset/MS_IR_H-NMR_Spectra_Database/25513513

[2] Training/Validation/Test dataset splitting: Including the split of real and null reactions for training, validation and test that can be directly fed into the model training and evaluation.

https://figshare.com/articles/dataset/Training_Validation_Test_set_split/25511056

[3] Model checkpoints: 

https://figshare.com/articles/dataset/Model_checkpoints/25513519

Detailed code usage scenario:

[1] Retrain the regular model: 1) install pytorch 1.12.1 (gpu version); 2) replace content in config_train.txt with correct file paths and select training mode; 3) run "python train_script.py -c config_train.txt" on gpu.

[2] Retrain the random drop model: Step 1) and 2) same as [1]; 3) run "python train_script_random_drop.py -c config_train.txt" on gpu; 4) 6 drop lists txt files (MS/IR/NMR for train/validation sets) will be generated under the current directory to indicate what examples are dropped with what spectral sources.

[3] Evaluate performance on regular model: 1) install pytorch 1.13.0 (cpu version) and rdkit 2020.09.1; 2) replace content in config_eval.txt with correct file paths and select evaluation mode; 3) run "python test_script.py -c config_eval.txt".

[4] Evaluate decisiveness behavior of model: step 1) same as [3]; 2) replace content in config_eval_decisive.txt with correct file paths and select evaluation mode; 3) run "python test_decisive.py -c config_eval_decisive.txt"; 4) 5 txt files will be created in the desiganted path reflecting the decoded SMILES as well as the decisive strings of reactants, MS, IR and H-NMR.

[5] Evaluate model performance with noised spectra: step 1) same as [3]; 2) replace content in config_eval_noise.txt with correct file paths, select evaluation mode, noise level and which spectral inputs are noised; 3) run "python test_noise.py -c config_eval_noise.txt".

[6] Evaluate model performance on solvent dataset: step 1) and 2) same as [3]; 3) run "python test_sol.py -c config_eval_sol.txt".

[7] Evaluate model performance on multi dataset: step 1) and 2) same as [3]; 3) uncomment section indicated in calc_acc.py to print out results before running the script; 4) run "python test_multi.py -c config_eval_multi.txt". 5) A .txt file will be generated with top5 smiles prediction generated (XXX means invalid smiles) ending with the ground truth smiles separated by ';'.

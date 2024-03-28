[1] Retrain the regular model: 1) install pytorch 1.12.1 (gpu version); 2) replace content in config_train.txt with correct file paths and select training mode; 3) run "python train_script.py -c config_train.txt" on gpu.   

[2] Retrain the random drop model: Step 1) and 2) same as [1]; 3) run "python train_script_random_drop.py -c config_train.txt" on gpu; 4) 6 drop lists txt files (MS/IR/NMR for train/validation sets) will be generated under the current directory to indicate what examples are dropped with what spectral sources.

[3] Evaluate performance on regular model: 1) install pytorch 1.13.0 (cpu version) and rdkit 2020.09.1; 2) replace content in config_eval.txt with correct file paths and select evaluation mode; 3) run "python test_script.py -c config_eval.txt".

[4] Evaluate decisiveness behavior of model: step 1) same as [3]; 2) replace content in config_eval_decisive.txt with correct file paths and select evaluation mode; 3) run "python test_decisive.py -c config_eval_decisive.txt"; 4) 5 txt files will be created in the desiganted path reflecting the decoded SMILES as well as the decisive strings of reactants, MS, IR and H-NMR.

[5] Evaluate model performance with noised spectra: step 1) same as [3]; 2) replace content in config_eval_noise.txt with correct file paths, select evaluation mode, noise level and which spectral inputs are noised; 3) run "python test_noise.py -c config_eval_noise.txt".

[6] Evaluate model performance on solvent dataset: step 1) and 2) same as [3]; 3) run "python test_sol.py -c config_eval_sol.txt".

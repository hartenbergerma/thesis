<img src="thesis_poster.png" alt="poster" width="1500">

## Dataset
The Hyperspectral data was provided by the [HSI Human Brain Database](https://hsibraindatabase.iuma.ulpgc.es/).
The dataset was acquired in three campaigns, totalling 71 Hyperspectral images from 39 patients.
Due to anomalies in some of the available image data, only a subset of the dataset was used.
Images from the third campaign were excluded as the corresponding dark references were seemingly not taken with the camera shutter closed.
Additionally, some images from the first campaign captured at a different hospital were not included because they had unusually high intensities. 
This results in a final set of 45 images from 25 patients:
<img src="dataset.png" alt="dataset" width="1000">

## Monte Carlo Simulations
The Monte Carlo Simulations of the tissue reflections are performed using [CUDAMCML](https://www.atomic.physics.lu.se/biophotonics/research/monte-carlo-simulations/gpu-monte-carlo/)

## Classification pipeline
#### hyperparameter tuning with [tuning.py](classification/tuning.py):
Example command:
CUDA_VISIBLE_DEVICES=1 python tuning.py \
    --mode baseline \ 		# model input mode
    --log_dir ./logs/bl  	# tensorboard logs and checkpoints
#### model training with [train.py](classification/train.py):
After the best hyperparameter configuration has been found the model configuration can be trained for each fold using the following command:

CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode baseline \				        # model input mode (new modes have to be added to train.py)
    --log_dir ./models/bl \		            # tensorboard logging dir and checkpoints
    --folds fold1 fold2 fold3 fold4 fold5 \	# folds on which models should be trained
    --hidden_dim 21 \				        # model hyperparameters...
    --last_layer_dim 17 \
    --num_layers 1 \
    --lr 0.000019785 \
    --weight_decay 0.00022366 \
    --batch_size 32
#### model testing with [test.py](classification/test.py)
calculates class prediction map for the test set of each of the five folds. Saves the classification performance metrics for each individual image. Metrics are saved under $log_dir/results

CUDA_VISIBLE_DEVICES=1 python test.py \
    --mode baseline \				        # model input mode (new modes have to be added to test.py)
    --log_dir ./models/bl \		            # logging directory from training step
    --folds fold1 fold2 fold3 fold4 fold5	# models to be tested (fold1 model was trained on fold 1)

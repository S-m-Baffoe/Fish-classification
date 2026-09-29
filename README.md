# Fish Classification

An exploratory image-classification project from the Runmila challenge, using underwater fish images and an ImageNet-pretrained Xception network. The notebook explores class imbalance, image augmentation, and transfer learning across 23 fish categories.

**Status:** This is a learning notebook with saved experimental outputs, not a finished prediction application. Some cells contain errors, and the model configuration needs revision before its accuracy can be interpreted as 23-class classification performance.

## Contents

- [GRP_1_Runmila_Challenge.ipynb](GRP_1_Runmila_Challenge.ipynb): dataset download, exploration, resampling, image generators, model construction, and an initial training run.
- `README.md`: project overview and guidance for working with the notebook.

The dataset and trained model weights are not included in this repository.

## Dataset

The notebook uses the [Fish4Knowledge recognition dataset](https://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/), described in its introduction as 27,370 underwater images across 23 categories. Class sizes are highly imbalanced. Images sharing a tracking identifier belong to the same fish trajectory.

The download and extraction cells retrieve `fishRecognition_GT.tar` and expect images under `/content/fish_image/`, with class directories such as `fish_01` through `fish_23`. The notebook's saved download log records an archive of approximately 487 MiB; dataset access depends on the external host.

## Open and explore

1. Download the notebook or clone this repository.
2. Open `GRP_1_Runmila_Challenge.ipynb` in Google Colab, or in a local Jupyter environment.
3. Review the cells before running. The notebook mounts Google Drive and downloads/extracts an external dataset.
4. Run the data-loading and exploration cells in order, adjusting filesystem paths for your environment.
5. Address the model and data-splitting issues below before running a training experiment.

The notebook imports Python packages including **NumPy, pandas, scikit-learn, OpenCV (`cv2`), Matplotlib, TensorFlow, and Keras**. Local use also requires Jupyter. The Google Drive mounting cell is specific to Colab and should be skipped or adapted locally.

No dependency lockfile is supplied. The notebook was created in 2021 and mixes `keras` and `tf.keras` imports, so a current environment may require import/API updates. A GPU can help with training, but the dense model head is large and may exceed available memory.

## Workflow

1. Download and extract the fish images.
2. Build a table of image paths and class labels; inspect image examples and class counts.
3. Explore repeated sampling to balance class counts and transformations such as rotation, shifts, zoom, shear, and horizontal flips.
4. Create training, validation, and test image generators.
5. Load Xception with ImageNet weights, freeze its base, and add dense layers.
6. Start model training and inspect the recorded output.

## Issues to resolve before evaluating results

- **Output/label mismatch:** the saved model ends with one sigmoid output and binary cross-entropy, while the generators produce 23 categorical labels. A 23-class experiment needs a compatible multiclass head and loss.
- **Data leakage risk:** one workflow balances the full table by duplicating images before a random train/test split. Split original data first and resample only the training set. Keep images from the same tracking identifier together when assessing generalization to unseen fish.
- **Saved execution errors:** augmentation examples report categorical-label type errors, and a balancing cell reports an undefined `train_df`. Some cells also spell `save_to_dir` as `save_t0_dir`.
- **Large model head:** the saved model summary contains approximately 436 million parameters, largely from flattening the feature maps before a 4,096-unit dense layer.
- **Incomplete evaluation:** saved training output is not a validated held-out benchmark. The notebook contains exploratory cells and outputs that may come from different execution orders.

This README documents the existing notebook; it does not modify or repair its code. Training and dataset downloads were not rerun when writing this documentation.

## Outputs and reuse

The notebook displays class summaries, example images, generator counts, a model summary, and training progress. There is no packaged command-line predictor or bundled trained model. To adapt the work, first establish a reproducible environment, correct the model/data issues above, and record a fresh evaluation on independent test data.

## Acknowledgments

The notebook identifies this work as the Runmila challenge and links to the Fish4Knowledge dataset. Refer to the dataset provider for its attribution and usage requirements. No repository license file is currently included.

# Investigating Transformer based U-Nets for clinicly significant prostate lesion segmentation in bpMRI

| T2                     | ADC                    | HBV                    |
|------------------------|------------------------|------------------------|
| ![t2w image ground truth (green) prediction (red)](/assets/10503_t2w.png "t2w image ground truth (green) prediction (red)") |  ![adc image ground truth (green) prediction (red)](/assets/10503_adc.png "adc image ground truth (green) prediction (red)")  |![hbv image ground truth (green) prediction (red)](/assets/10503_hbv.png "hbv image ground truth (green) prediction (red)")  |

## introduction
The code from my Bachelor-Thesis on prostate lesion segmentations.

With this code I compared 3 models (Basic UNet, SwinUNETR and a ResNet based UNet) on the PIC-AI dataset. It contains code for downloading and preprocessing the dataset, the implementation of a training process including parameter search via grid-search and code for the evaluation with dice-scores, average precision and the metric from the PIC-AI challange. As well as the possibility to save the predictions as NIFTI-files to view them in programs like 3D Slicer. 

## getting started
### requirements
python version = 3.10
install all requirements with
```
pip install -r requirements.txt
```
To read the image data with monai we also need an suitable reader. The easiest way is to just install all dependencies of monai:
```
pip install 'monai[all]'
```

### download PIC-AI Dataset
For downloading the dataset the script *download.py* in data can be used.
```
python -m data.download
```
This process can take a while because it needs to download ~25GB of images.

### preprocess data
before the training the image data were preprocessed. This process contains a normalization, a resampling to the shape
of the t2w-images and resampling on a spacing of 0.3 x 0.3 x 1.0mm. And a cutting out of the prostate region based on 
the prostate segmentation. This process will take quite some time (~3 hours).

**Attention: because of the resampling in this process the dataset becomes much bigger!**
```
python -m src.preprocess
```

### train model
To train a model the train function can be used.
```
python -m src.train -n testmodel -m basic -e 20
```
There are multiple options:

| flag                  | effect                                                      |
|-----------------------|-------------------------------------------------------------|
| --model \ -m          | define the model to train ("basic", "resnet", "transformer" |
| --epochs \ -e         | number of epochs to train                                   |
| --name \ -n           | name of the model                                           |
| --loss \ -l           | loss function to use (dice, diceCE, diceFocal)              |
| --loss \ -l           | loss function to use (dice, diceCE, diceFocal)              |
| --batch-size \ -b     | batch size                                                  |
| --learning-rate \ -lr | learning rate                                               |
| --weight_decay \ -wd  | weight decay                                                |
| --gamma \ -g          | gamma for exponential lr scheduler                          |
| --ce_weights \ -cw    | weights for classes in CE loss                              |
| --cuda-device \ -c    | cuda device (int)                                           |
| --worker \ -w         | number of workers for Dataloader                            |
| --dataset \ -d        | dataset to train on (picai, inhouse)                        |


### predict MRI data
To predict lesions from a patients bpMRI images the predict.py script can be used, by calling this command:
```
python -m src.predict --source path\to\images
```
There are multiple options:


| flag               | effect                                        |
|--------------------|-------------------------------------------------|
| --multi \ -m       | predicting multiple images at once              |
| --extra_folder \ -e | creating an extra folder for the prediction     |
| --post_fix \ -p    | adding a special postfix to the prediction file |


## References
For the evaluation I use the code from the PIC-AI Challenge:

● [A. Saha, J. J. Twilt, J. S. Bosma, B. van Ginneken, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, "Artificial Intelligence and Radiologists at Prostate Cancer Detection in MRI: The PI-CAI Challenge (Study Protocol)", DOI: 10.5281/zenodo.6667655](https://zenodo.org/record/6667655)
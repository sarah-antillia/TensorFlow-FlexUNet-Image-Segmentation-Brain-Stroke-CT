<h2>TensorFlow-FlexUNet-Image-Segmentation-Brain-Stroke-CT (2025/09/19)</h2>

Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for <b>Brain Stroke CT (Bleeding and Ischemia) </b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels PNG dataset with colorized masks (Bleeding:red, Ischemia:green) which was derived by us from 
<a href="https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset/data">
<b>Brain Stroke CT Dataset</b><br>
CT Image Dataset for Brain Stroke Classification, Segmentation and Detection
</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<b>rgb_map (Bleeding:red, Ischemia:green)</b> <br>
<br>
<table>
<tr>

<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/seg_result_header.png" width="960" height="auto"></td>
<!--

<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
 -->
</tr>

<tr>

<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/summary_seg_result.png" width="960" height="auto"></td>


<!--
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Bleeding_16818.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Bleeding_16818.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Bleeding_16818.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Bleeding_16962.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Bleeding_16962.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Bleeding_16962.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Ischemia_10298.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Ischemia_10298.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Ischemia_10298.png" width="320" height="auto"></td>

</tr>
-->
</table>

<hr>
<br>

<h3>1 Dataset Citation</h3>
The dataset used here was obtained from
<br><br>
<a href="https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset/data">
<b>Brain Stroke CT Dataset</b><br>
CT Image Dataset for Brain Stroke Classification, Segmentation and Detection
</a>
<br>
<h4>About Dataset</h4>
<b>Description</b><br>
The stroke dataset, which includes head computed tomography (CT) cross-sectional images, 
was prepared within the scope of the Artificial Intelligence in Healthcare competition 
held in Istanbul in 2021 with the support of the Ministry of Health, 
General Directorate of Health Information Systems, under the management of the Turkish Health Institutes (TUSEB).
<br>
Dataset includes images of 6653 CT brain slices with the following distribution:
<br>
<li>No stroke: 4428 images</li>
<li>Ischemia: 1131 images</li>
<li>Bleeding: 1094 images</li>

<h4>Materials and Methods</h4>

Dataset recorded over the period covering 2019 and 2020 were centrally screened from the e-Pulse and 
Teleradiology System of the Republic of Türkiye, Ministry of Health using various codes and filtering criteria. 
The data set was anonymized. The data set was prepared, pooled, curated, and annotated by 7 radiologists.

<h4>Citation</h4>
Koç U, Akçapınar Sezer E, Alper Özkaya Y, et al.
<a href="https://www.eajm.org/en/artificial-intelligence-in-healthcare-competition-teknofest-2021-stroke-data-set-133405">
 Artificial intelligence in healthcare competition (TEKNOFEST-2021): 
</a>
 <br>
Stroke data set. Eurasian J Med., 2022;54(3):248-258.<br>

<h4>LICENSE</h4>
The data is collected from https://acikveri.saglik.gov.tr/Home/DataSetDetail/1 and merged into a single folder for ease of use.
<br>
All legal rights belong to the Republic of Türkiye Ministry of Health.
<br>
<br>
<h3>
2 Brain-Stroke-CT ImageMask Dataset
</h3>
<h3>2.1 ImageMask Dataset Generation</h3>
 If you would like to train this Brain-Stroke-CT Segmentation model by yourself,
 you have to generate the PNG dataset by using the following Python scripts:
 <br><br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>

Please put Brain_Stroke_CT_Dataset which was downloaded from the kaggle website under <b>./genrator</b> folder.<br>
<pre>
./generetor
 ├─ImageMaskDatasetGeneratory.py
 ├─split_master.py
 └─Brain_Stroke_CT_Dataset    # Downloaded from the kaggle website 
      ├─Bleeding
      │  ├─DICOM
      │  ├─OVERLAY
      │  └─PNG
      ├─External_Test
      │  ├─DICOM
      │  ├─MASKS
      │  ├─OVERLAY
      │  └─PNG
      ├─Ischemia
      │  ├─DICOM
      │  ├─OVERLAY
      │  └─PNG
      └─Normal
          ├─DICOM
          └─PNG
</pre>
<br>
 The first generator script generates a master image and mask dataset
of 512x512 pixels PNG format from <b>OVERLAY</b> and <b>PNG</b>
 in <b>Bleeding</b> and <b>Ischemia</b> under Brain_Stroke_CT_Dataset folder.
<br>
Each mask is generated from a PNG file in the OVERLAY folder and the corresponding PNG file in the PNG folder,
by subracting the PNG image from the OVERLAY image, and colorizing it with color_map (Bleeding:red, Ischemia:green).
<br><br>
 The second splitter simply splits the master into test, train and valid subsets. <br>

By running these Python scripts, finally a 512x512 pixels PNG Brain-Stroke-CT dataset will be 
created under <b>dataset</b> folder as shown below.<b></b>.  

<pre>
./dataset
└─Brain-Stroke-CT
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Brain-Stroke-CT Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/Brain-Stroke-CT_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<b>You may not redistribute this training dataset generated from 
<a href="https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset/data">Brain_Stroke_CT_Dataset</a>, and 
commercial use of this dataset is prohibited.</b> <br>
<br>

<h3>2.2 Mini-test Dataset Generation</h3>
You also have to generate a <b>mini_test</b> dataset from <b>Brain-Stroke-CT/test</b> subset by yourself, 
depending on your choices,
which can be used in EpochChangeInferencer Callback and an actual segmentation (predictioin) process.
<br>
<pre>
./projects
└─TensorflowFlexUNet
       └─Brain-Stroke-CT
             └─mini_test
                ├─images
                └─masks
</pre>
<!--
<b>You may not redistribute your own mini_test dataset.</b><br>
-->
<br>
<br>
<h3>2.3 Tran Images and Masks Sample </h3>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We have trained Brain-Stroke-CT TensorFlowUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Brain-Stroke-CTand, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowUNet.py">TensorFlowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (3,3)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Brain-Stroke-CT 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"

;Brain-Stroke CT
; rgb color map dict for 3 classes.
;                 Bleeding:red, Ischemia:green
rgb_map = {(0,0,0):0,(255,0,0):1,(0,255,0):2,}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 24,25,26)</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 49,50,51)</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 51 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/train_console_output_at_epoch51.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Brain-Stroke-CT</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Brain-Stroke-CT.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/evaluate_console_output_at_epoch51.png" width="720" height="auto">
<br><br>Image-Segmentation-Brain-Stroke-CT

<a href="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Brain-Stroke-CT/test was low, but dice_coef high as shown below.
<br>
<pre>
categorical_crossentropy,0.0248
dice_coef_multiclass,0.9889
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Brain-Stroke-CT</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for Brain-Stroke-CT.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 512x512 pixels </b><br>
As shown below, this segmentation model failed to detect some Bleeding and Ischemia lesions.<br>

<b>rgb_map (Bleeding:red, Ischemia:green)</b> <br>
<br>
<table>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/seg_result_header.png" width="960" height="auto"></td>

<!--
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>

</tr>
-->

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/seg_result_1.png" width="960" height="auto"></td>

</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/asset/seg_result_2.png" width="960" height="auto"></td>

</tr>

 
<!--

<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Bleeding_16818.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Bleeding_16818.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Bleeding_16818.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/XBleeding_10050.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/XBleeding_10050.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/XBleeding_10050.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/XBleeding_10788.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/XBleeding_10788.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/XBleeding_10788.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Ischemia_10238.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Ischemia_10238.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Ischemia_10238.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Ischemia_10354.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Ischemia_10354.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Ischemia_10354.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/images/Ischemia_10650.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test/masks/Ischemia_10650.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Brain-Stroke-CT/mini_test_output/Ischemia_10650.png" width="320" height="auto"></td>
</tr>
-->>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Artificial Intelligence in Healthcare Competition (TEKNOFEST-2021):Stroke Data Set</b><br>
Ural Koç,, Ebru Akçapınar Sezer, Yaşar Alper Özkaya, Yasin Yarbay4 , Onur Taydaş,<br>
Veysel Atilla Ayyıldız , Hüseyin Alper Kızıloğlu, Uğur Kesimal, İmran Çankaya, Muhammed Said Beşler,<br>
Emrah Karakaş, Fatih Karademir, Nihat Barış Sebik, Murat Bahadır , Özgür Sezer,<br>
Batuhan Yeşilyurt, Songul Varlı, Erhan Akdoğan, Mustafa Mahir Ülgü , Şuayip Birinci<br>

<a href="https://www.eajm.org/en/artificial-intelligence-in-healthcare-competition-teknofest-2021-stroke-data-set-1618971">
https://www.eajm.org/en/artificial-intelligence-in-healthcare-competition-teknofest-2021-stroke-data-set-1618971
</a>

<br><br>
<b>2. Hemorrhagic stroke lesion segmentation using a 3D U-Net with squeeze-and-excitation blocks</b><br>
Valeriia Abramova, Albert Clèrigues, Ana Quiles, Deysi Garcia Figueredo, Yolanda Silva, Salvador Pedraza,<br>
 Arnau Oliver, Xavier Lladó<br>
<a href="https://www.sciencedirect.com/science/article/pii/S0895611121000574">
https://www.sciencedirect.com/science/article/pii/S0895611121000574
</a>
<br>
<br>
<b>3. Segmentation of acute stroke infarct core using image-level labels on CT-angiography</b><br>
Luca Giancardo, Arash Niktabe, Laura Ocasio, Rania Abdelkhaleq, Sergio Salazar-Marioni, Sunil A Sheth<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10011814/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC10011814/
</a>
<br>
<br>
<b>4.Segmenting Small Stroke Lesions with Novel Labeling Strategies</b><br>
Liang Shang, Zhengyang Lou, Andrew L. Alexander, Vivek Prabhakaran,<br>
William A. Sethares, Veena A. Nair, and Nagesh Adluru<br>
<a href="https://arxiv.org/pdf/2408.02929">
https://arxiv.org/pdf/2408.02929
</a>


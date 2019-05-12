# Facial Expression Recognition Using Deep Learning 
Divya Periyakoil, Isabella Maceda, John Xinyu Lin, Brandon Scolieri

## Abstract
Facial emotion recognition (FER), the problem of recognizing emotion from facial images, is an important problem with many real-world applications. We construct a convolutional neural network with residual layers to categorize facial images into eleven emotion categories, trained on images from the AffectNet database. Our best model currently has validation accuracy of around 35%, and we hope that with better computational resources, and with techniques to combat model overfitting, we may be able to reach state-of-the-art performances.

## Background (problem statement, our project's goal, etc.) (John and Brandon)

Facial emotion recognition (FER), the ability to recognize human emotion expressed by faces, has been, and still is, an important yet challenging problem in computer vision and artificial intelligence. For instance, being able to recognize human emotion has significant implications for robotics, especially human-robot interaction. Because non-verbal expressions can account for around two-thirds of human communication [4], having FER abilities would facilitate richer interactions between humans and machines. Other uses for FER include augmented and virtual reality [4], as well as labeling emotions for people with autism, who may otherwise have trouble interpreting other facial expressions on their own. Hence, developing automatically systems for FER is crucial, not only for academic ambitions, but for practical purposes as well. 
With the advent of better computational resources and image data in recent decades, researchers have been examining and developing neural-network approaches toward FER [2]. Unlike traditional approaches, which usually hand-craft facial image features [4], deep learning approaches instead learn these features via neural networks, often providing improved performance, especially on novel images outside training data [2]. Usually, this is because traditional models are tailored toward facial images taken in a fixed, controlled environment, and are not flexible enough to adapt to other images from other environments (including other controlled environments as well as uncontrolled, "in the wild" images) [2]. State-of-the-art algorithms using neural network approaches have generally good results, with some models achieving over 90% accuracy on controlled-environment facial databases, and others achieving around 80% on in-the-wild images [1].
Our goal is to design an FER system, using deep learning, that can identify the emotions represented in images of human faces. Specifically, we use static images of faces, and assign a label to each image corresponding to the emotion that face represents. This uses the categorical model of facial expression quantification [1]. Other models include the valence and arousal model, which measures emotions on a continuous scale based on their positivity/negativity and excitement/calmness [1]. However, for simplicity, we decided to focus on categorizing emotions into discrete classes. as a metric of our performance, we mainly use accuracy on held-out data (specifically, validation data, as our dataset did not come with test data) to assess our model.

## Materials, Methods, Procedure, etc. (Divya and Isabella)

To train our model, we wanted to use a large database of facial images, preferably one with diverse, high-quality images. Hence, wWe decided to used the AffectNet dataset, which containsis comprised of approximately 420,000 labeled in-the-wild photos that are classified into the following 11 categories: neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, and non-face. The imagesdata were obtained byfrom the AffectNet team searching Google, Bing, and Yahoo for queries in six different languages: English, Spanish, Portuguese, German, Arabic, and Farsi [1]. Out of the 1,000,000 plus images gathered, around 420,000 of them were manually annotated by a team of hired annotators [1].  
We decided to use Google Drive and Google Colaboratory (Colab) for our project, because it had the storage capacity that we needed for our project. After downloading the AffectNet database, we uploaded our files to a shared Google Drive folder, and used Colab notebooks to perform all our data processing and model training, saving our results in Google Drive.
The AffectNet imagesphotos were accompanied by CSVcsv files containing the file path for each image, class, and other information such as valence and arousal, which we did not use forwere not utilized in this project. The csv labels were manual annotations of each of the images by the AffectNet team. Using these labels as the ground truth, we trained our model and tested our models’ outputs comparatively. 

Given our memory limitations Next, we had to preprocess the images to make them of uniform size, and of dimensions that could be handled withigiven our memory limitations. Our dataset consistsconsisted of over 400,000 images, so we also had to account for the computation time of preprocessing the images. We used PIL (Python Image Library) to resize each of the images to dimensions 32x32. We then converted each image into a tensor of dimensions 32x32x3, with 32x32 representing the dimensions of the image, and 3 representing the number of color channels. We used multithreading to ameliorate the preprocessing time.
The AffectNet dataset also included ResNeXt (aggregated residual) networks for annotating images under the categorical model (with the 11 aforementioned categories) and the valence and arousal model; these models were used to automatically annotate the remaining 550,000 plus images (which we did not use). We took inspiration from their categorical model when coming up with our own model, but otherwise did not base our model off their architecture. However, we did train their model, with slight modifications, for comparison purposes, which we describe in our concluding discussion.


## Various different iterations of the models, what worked and what didn’t, results (Divya and Isabella)
WIn the process of creating our final model, we created over 10 different models before creating our final model. The models listed below show the progression of how we tweaked the intermediate models and arrived at our final model. Each model was implemented from scratch using the machine learning libraries Keras and Tensorflow and used a batch size of 128 and had an input shape of 128x32x32x3.

The first model we triedapplied was a simplesimplistic convolutional neural network comprised of convolutions of sizesized 32, 64, 128 and 256; max pooling; and a dense layer of dimension 512. We noticed that the model was severely overfitting ourthe data, which was made evident fromby our high training accuracy and low validation accuracy. After approximately 15 to 20 epochs, the validation accuracy plateaued at around 25%. 
Using the stacked convolution model as our baseline, we augmented this initial model further in our next iteration by adding 5 residual layers of dimension 256 after the convolution stack. Our motivation for adding residual layers was to counter the overfitting we saw in our first benchmark model. Each residual layer consisted of a linear activation, convolutional layers, a ReLUrelu activation, and finallying an addition layer. The input and output sizes were of 128x32x32x256. This improved our validation accuracy to 29.7%. We also attempted to interweave residual and convolutional layers but found that doing so led to increased overfitting. 
In our third iteration, we noted the severe imbalance of the number of photos belonging to each class. For example, the "happy category," with 125,734 images, had nearly 30 times more photos than the "disgust category," which only had 3,676 photos. This presented issues while training the model, as we sampled 5000 images per epoch. Thus, we decided to sample max(5000, all of the images) from each category and make this the training set instead. This made the training set more balanced and fair toward every category in the dataset. As a result, wWe saw a considerable increase in our validation accuracy, to 35.6%. 

## Visualizations (John and Brandon)

Figure 1. Accuracies for our basic model for the first 100 epochs. Notice how validation accuracy tends to hover around 25% after 20 epochs. Training accuracy continues to rise, indicating clear overfitting.

Figure 2. Accuracies for the second model for the first 25 epochs, approaching 30% validation accuracy. While we have not shown all epochs, already we notice plateauing in validation accuracy.

Figure 3. An overview of our architecture for the second and third iterations. Blue rectangular prisms represent regular convolutional layers.

Figure 4. Examples of preprocessed images, scaled down to 32x32 pixels.
## Challenges/What went wrong and why
The first challenge we encountered in pursuing our project was the time to download, extract and upload the data. Doing so took longer than we expected, which delayed the amount of time and resources left for processing and training our models. Another challenge that our team faced was Google Colab’s RAM limitation. This limited the number of epochs we could run and the number of steps per epoch that we could use. 
One of our worst performing models, placed a ReLUrelu activation at the end of our model, causing an inordinate amount of overfitting because it made our gradients vanish for all negative inputs.

## Future Questions to Ask, applications of our model, changes that we would like to implement to improve our model (Divya and Isabella)
In the future, we hope to improve our model by using some of the following methods: whitening, batch normalization (a more effective use of batch normalization), layer normalization, and gated tanh units. 

## Conclusion (John and Brandon)
As we have learned, designing a neural network that accurately categorizes facial emotions is not an easy task, especially given our computational constraints. While we have not reached accuracies comparable to other state-of-the-art algorithms (from our introductory discussion), we have learned the following lessons:
Using residual layers helps increase our model performance. First espoused by He et al. in their paper “Deep Residual Learning for Image Recognition,” residual connections have enabled deeper architectures that have demonstrated better performance compared to their non-residual counterparts [3]. By mitigating the training accuracy degradation (described in He et al.'s experiments) that usually occurs with an increased number of layers, deeper models become faster to train, and result in better accuracies overall [3]. Hence, we decided to add residual layers to give our network more representation power while preserving accuracy.
Our dataset can significantly affect our overall performance. Initially, we had ignored the severe imbalance among our training images, and simply decided to train our model anyway. However, as two of our graduate student instructors pointed out (see acknowledgements), doing so can adversely affect our model's performance, especially if it learns to favor one class over another. With that in mind, we decided to make the changes mentioned above for our third major model iteration.
With better computing resources, our model may be promising. Apart from developing our own model, we ran the ResNeXt model provided with the AffectNet dataset, only changing the network's image input size to 32x32x3 (as we have downscaled our images, as mentioned above). Running the model for around 15 epochs gives us a validation accuracy of around 25%, comparable to our first model iteration, and worse than our second and third iterations (respectively around 30% and 35%). The AffectNet documentation states that their ResNeXt model has an average accuracy of 65%; however, that accuracy is over the span of 200 epochs. With more training time and memory, perhaps our model can also reach similar accuracies, if not greater.

## Team Contributions
Teammate: Sections contributed | % Contributed
Divya Periyakoil: Research, Modeling, Poster, Report, Visualizations | 25%
Brandon Scolieri: Research, Modeling, Poster, Report, Visualizations | 25%
Isabella Maceda: Research, Modeling, Poster, Report, Visualizations  | 25%
John Xinyu Lin: Research, Modeling, Poster, Report, Visualizations   | 25%

## Acknowledgements
We would like to acknowledge David Chan, our advising graduate student instructor, for guiding us through  our project. Additionally, we would like to recognize Professor John Canny for teaching the course and givinge us guidance on how to improve our model’s performance, as well as Phillipe Laban, a graduate student instructor, who gave us instrumental advice to improveameliorate our model during our poster session. 

## Bibliography
[1] A. Mollahosseini; B. Hasani; M. H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild," in IEEE Transactions on Affective Computing, 2017.

[2] A. Mollahosseini, D. Chan and M. H. Mahoor, "Going deeper in facial expression recognition using deep neural networks," 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Placid, NY, 2016, pp. 1-10. doi: 10.1109/WACV.2016.7477450

[3] He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.

[4] Ko, Byoung Chul. “A Brief Review of Facial Emotion Recognition Based on Visual Information.” Sensors (Basel, Switzerland) vol. 18,2 401. 30 Jan. 2018, doi:10.3390/s18020401


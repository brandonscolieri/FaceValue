# Facial Expression Recognition Using Deep Learning 
John Xinyu Lin, Isabella Maceda, Divya Periyakoil, Brandon Scolieri

## Abstract
Affect is a term that describes the outward expression of emotion and feelings [1]. Facial expression recognition (FER) or facial affect recognition (FAR) is the process of classifying human affect in facial image data. Automated affective computing achieved through FER is a challenging problem in the field of image classification due to the high variability of in-the-wild images. A subset of the factors that contribute to the complexity of this problem include variations in: image dimension, photograph angle, image lighting, background image data, and the human ability to express a wide range of facial expressions for any given categorical affect. Due to the multi-dimensional nature of this problem, we chose to approach this challenge by implementing a deep neural network architecture that leverages the power of deep learning to automatically capture the highly nuanced features of facial image data.

## Background

Facial expression recognition continues to be an important and persistent problem in the fields of computer vision and artificial intelligence; with significant applications in robotics (especially human-robot interaction), treatment of neurological disabilities, sentiment analysis, and more. Furthermore, considering that approximately two-thirds of human communication occurs through non-verbal expressions [4], having the ability to accurately assess image data with FER would facilitate significantly more robust interactions between humans and machines. Some practical examples of systems that could utilize this functionality include augmented reality systems and automated learning systems that label emotions for people affected by disabilities such as Autism; which can impair an individual’s ability to recognize facial emotions on their own. With the advent of better computational resources and image data in recent decades, researchers have been examining and developing neural-network approaches toward FER [2]. Unlike traditional approaches, which often utilize a discrete and relatively simple set of hand-crafted facial image features [4], deep learning approaches instead learn these features via neural networks. This approach often yields improved performance, especially on novel and in-the-wild image data [2]. This improvement on in-the-wild image classification using DNN’s can largely be attributed to the fact that traditional models lack the necessary feature complexity and are tailored toward facial images taken in controlled environments. These characteristics render inelastic networks that cannot adequately generalize and adapt to other images from other environments (including other controlled environments as well as uncontrolled, (i.e. “in-the-wild”) images) [2]. State-of-the-art algorithms using neural network approaches have generally good results, with some models achieving over 90% accuracy on controlled-environment facial databases, and others achieving around 82% on in-the-wild images [1].  

Our goal was to design a FER system, using deep learning, that could identify the affect present in images of human faces. To accomplish this we began by constructing a convolutional neural network with residual layers to categorize static in-the-wild images of faces from the AffectNet database into eleven discrete affect categories.  

While some models include continuous features such as valence and arousal, which quantify levels of sentiment (i.e. positivity/negativity of an image) and intensity of response (i.e. excitement/calmness of an image) respectively [1], we chose to allow the network to learn these features on its own and focus on categorizing emotions into discrete classes. To measure our performance, we assessed model accuracy on held-out validation data.  

## AffectNet and Data Processing  

The original AffectNet dataset is comprised of approximately 1,000,000 in-the-wild facial images that were gathered by querying Google, Bing, and Yahoo using 1,250 emotion related tags in English, Spanish, Portuguese, German, Arabic, and Farsi. About half of the original dataset was manually categorized into the following eleven discrete categories: neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, and non-face. The data was obtained directly from the AffectNet team. The image dimensions vary and the distribution of images by category was not uniform. In addition, the dataset contained .csv files that describe the file path, class label, and some information that was not used for this project such as, valence and arousal. 

Due to computational limitations we used a manually labeled subset of approximately 394,000 facial images, which were sampled from the complete AffectNet dataset. The labels from the manually annotated images served as the ground truth for our models. For housing the data we decided to use Google Drive since it provided sufficient storage capacity for free. Since we were housing the data using Google Drive we used Google Collaboratory’s access to free GPU resources as a means for performing and accelerating the training computation. We found that there are significant drawbacks to using both of these free services, which will be elaborated on in the *Project Challenges and Takeaways* section.  
 
After obtaining the data, we preprocessed the images to scale them down to uniform dimensions that were within the parameters of the memory limitations imposed by Google Collaboratory. Despite using only a fraction of the dataset, with over 394,000 images we still had to account for the computation time required to preprocess the images. To complete the preprocessing phase we used the Python Image Library (PIL) to resize each of the images to dimensions of 32x32. Following that, each image was converted into a tensor of dimensions 32x32x3, where 32x32 represents the image dimensions in pixels, and 3 represents the number of color channels. We utilized multithreading to ameliorate the preprocessing time.


| Labeled Affect Category | Neutral | Happy   | Sad   | Surprise | Fear  | Disgust | Anger  | Contempt | None   | Uncertain | Non-face |
|------------------------:|---------|---------|-------|----------|-------|---------|--------|----------|--------|-----------|----------|
| Number of images        | 71,053  | 125,734 | 24658 | 13,512   | 6,211 | 3,676   | 23,872 | 3,530    | 31,769 | 11,286    | 78,751   |

*Figure 1. Distribution of images by category. Total images: 394,052*



![Sample downsized images](images/sample-images.png)
*Figure 2. Examples of preprocessed images, scaled down to 32x32 pixels.*


## Model Architectures

In the process of creating our most successful solution, we created over 10 different models. In order to illustrate the progression of our work the models below were selected to represent our baseline, intermediate, and final model. Each model used a batch size of 128, had an input shape of 128x32x32x3, and was evaluated by computing the categorical cross-entropy loss.

![Cross-entropy](more-imgs/small-cross-entropy.jpg)

*Figure 3. General cross-entropy loss equation.*

1.	The first model we created was a simplistic convolutional neural network comprised of convolutions with sizes 32, 64, 128 and 256; max pooling; and a dense layer of dimension 256. We noticed that the model was severely overfitting the data, which was made evident by our high training accuracy and low validation accuracy. After approximately 15 to 20 epochs the validation accuracy plateaued at around 25%.  

    ![First model iteration architecture](more-imgs/model-iter1.jpg)
    *Figure 4. An overview of our architecture for the first iteration. Blue rectangular prisms represent regular convolutional layers.*

2.	Using the stacked convolution model as our baseline we augmented this initial model further in our next iteration by adding 5 residual layers of dimension 256 after the convolution stack. Our motivation for adding residual layers was to counter the overfitting we saw in our initial model. Each residual layer consisted of a linear activation, convolutional layers, a ReLU activation, and a summation layer. The input and output sizes were of 128x32x32x256. This raised our validation accuracy to 29.7%. We also attempted to interweave residual and convolutional layers, but found that this approach led to increased overfitting.

	![Residual layer](more-imgs/residual.jpg)
    *Figure 5. An overview of our residual layer.*

3.	In our third iteration, we corrected the severe imbalance of the number of photos belonging to each class. For example, the happy category had nearly 30 times more photos than the disgust category. This presented issues while training the model, as we sampled 5000 images per epoch. Thus, in order to balance the training set we decided to sample max(10,000, all of the images) from each category and use this as our new training set. In addition, we normalized our images (centered them) and upon testing saw a considerable increase in our validation accuracy, to 36.32%. 

    ![Second and third model iterations architecture](more-imgs/model3.jpg)
    *Figure 6. An overview of our architecture for the second and third iterations. Blue rectangular prisms represent regular convolutional layers.*


### Experimentation Notes  

The AffectNet team used a ResNeXt model for labeling the remaining images that were not manually annotated. To ensure that we were training on the most accurate data we did not use the images labeled by ResNeXt; however, we did run some preliminary experiments using a modified version of ResNeXt. Our modifications included altering the acceptable input dimensions to 32x32, adding additional batch normalization, adding dropout to help counteract overfitting, adding an RMSProp optimizer, and tuning various hyperparameters such as the learning and decay rates. Ultimately this model required computational resources that were beyond the scope of the free resources we were using and we frequently ran into timeout issues resulting from surpassing Google Collaboratory’s RAM limitations. This limited experimentation with this model and we opted to create an approach of our own instead of pursuing further development of the ResNeXt model.  


## Results

#### Loss and Accuracy on Training and Validation Sets for Model 1
![First model iteration accuracy](images/model-iteration1-loss.png)
*Figure 7. Losses for our basic model for the first 100 epochs. Training loss continues to go down but validation loss does not.*

![First model iteration accuracy](images/model-iteration1.png)
*Figure 8. Accuracies for our basic model for the first 100 epochs. Validation accuracy tends to hover around 25% after 20 epochs. Training accuracy continues to rise, indicating clear overfitting.* 


#### Loss and Accuracy on Training and Validation Sets for Model 2
![First model iteration accuracy](images/model-iteration2-loss.png)
*Figure 9. Losses for our second model for the first 25 epochs. Both training and validation losses seem to stabilize, but validation loss shows more variance.*

![Second model iteration accuracy](images/model-iteration2.png)
*Figure 10. Accuracies for the second model for the first 25 epochs. Validation accuracy plateaus around 30% after 25 epochs.*  


#### Loss and Accuracy on Training and Validation Sets for Model 3  
![First model iteration accuracy](images/model-iteration3-loss.png)
*Figure 11. Losses for our third model for 10 epochs.*

![Second model iteration accuracy](images/model-iteration3.png)
*Figure 12. Accuracies for the third model for 10 epochs.*  



## Project Challenges and Takeaways
As with any ambitious project, the challenges are often numerous and novel, and our project was no different. The very first obstacle we encountered was Google Drive’s upload speeds. Although using Google Drive to house our data was free, the upload time associated with this service nearly rendered it entirely unfeasible for a dataset of our size. Another challenge that our team faced was Google Collaboratory’s RAM limitation. This limited the number of epochs we could run, the number of steps per epoch that we could use, the image size we could train on, and effectively lowered the amount of deep experimentation we could perform. Overall we concluded that when it comes to a project of this caliber, having access to reliable GPU resources for hardware acceleration would have a significant impact on the project as a whole. Aside from resource limitations, we frequently ran into the challenge of counteracting overfitting. One example of a correction we made to our code happened when we found that one of our worst performing models contained a ReLU activation at the end. This effectively made our gradients vanish for all negative inputs and produced an inordinate amount of overfitting.  

From our experience with this challenging problem, some valuable insights that we have gained include the power of utilizing residual layers, the aforementioned importance of reliable computational resources, and the necessity of using a balanced data set.  With regard to residual layers, using residual layers significantly improved model performance. First espoused by He et al. in their paper *Deep Residual Learning for Image Recognition*, residual connections have enabled deeper architectures that have demonstrated better performance compared to their non-residual counterparts [3]. By mitigating the training accuracy degradation (described in He et al.'s experiments) that usually occurs with an increased number of layers, deeper models become faster to train, and result in better accuracies overall [3]. Hence, by employing this method we were able to add residual layers that gave our network more representational power whilst preserving accuracy. Finally in regard to using a balanced data set, this is an absolute necessity since training on imbalanced data can produce a network that is inherently more likely to produce a given label simply because the label is more prevalent in the dataset. We observed the effects of this before and after balancing our training data, and correction of this resulted in performance improvements. 

## Moving Forward  
In work to come we intend to improve our model by implementing whitening, layer normalization, and gated tanh units. We also plan on pursuing different resources such as AWS instead of Google applications. With these future modifications we will train using a larger image size (even though down sampling may render the effects of larger image sizes to be minor), implement the aforementioned modifications to our model, and train for longer.  

## Team Contributions
Teammate: Sections contributed | % Contributed  

Divya Periyakoil: Research, Modeling, Poster, Report, Visualizations | 25%  

Brandon Scolieri: Research, Modeling, Poster, Report, Visualizations | 25%  

Isabella Maceda: Research, Modeling, Poster, Report, Visualizations  | 25%  

John Xinyu Lin: Research, Modeling, Poster, Report, Visualizations   | 25%  

## Acknowledgements
We would like to acknowledge David Chan, our advising graduate student instructor, for guiding us throughout our project. Additionally, we would like to recognize Professor John Canny for teaching the course and giving us guidance on how to improve our model’s performance, as well as Phillipe Laban, a graduate student instructor, who gave us instrumental advice to ameliorate our model during our poster session. We would also like to thank the AffectNet team for allowing us access to their dataset so that we could work on this project. 

## Bibliography
[1] A. Mollahosseini; B. Hasani; M. H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild," in IEEE Transactions on Affective Computing, 2017.

[2] A. Mollahosseini, D. Chan and M. H. Mahoor, "Going deeper in facial expression recognition using deep neural networks," 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), Lake Placid, NY, 2016, pp. 1-10. doi: 10.1109/WACV.2016.7477450

[3] He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 770-778.

[4] Ko, Byoung Chul. “A Brief Review of Facial Emotion Recognition Based on Visual Information.” Sensors (Basel, Switzerland) vol. 18,2 401. 30 Jan. 2018, doi:10.3390/s18020401


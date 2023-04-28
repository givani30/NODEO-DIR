---
title: 'Reproducibility of NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration '
disqus: hackmd
---

Reproducibility of NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration
===
Authors:
* Givani Boekestijn (4710193)
* Thomas van Os (4686802)
* Han Zhang (5782449)

##    Introduction
Deformable image registration (DIR) is a fundamental problem in medical image analysis that involves establishing spatial correspondences between images. Traditionally, DIR is solved as a pair-wise optimization problem, and many existing deep learning-based registration approaches rely on supervised learning from labeled image pairs. However, there are limitations to these methods, including the inability to incorporate additional assumptions or constraints on the solution. In this context, a recent [paper](https://arxiv.org/abs/2108.03443) proposes a novel approach to DIR based on neural ordinary differential equations (NODEs). By imposing loss penalties on the intermediate states of the trajectory, the proposed framework allows for greater flexibility in the type and number of assumptions that can be imposed on the solution, and can be extended to multiple-image sets by adding intermediate supervision. This paper presents a comprehensive overview of the proposed framework, as well as illustrative examples demonstrating its properties and capabilities on 2D and 3D images.

The goal of this blog post is to verify and present our results in reproducing various aspects in the paper: “NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration”. To be more specific, first, we reproduce the table 1 in orignal paper with standard OASIS dataset, Next, we will modify the code to enable the application of algorithm to a 2D image. Last, our focus is to replicate the visualization of the warped moving image and deformation field grid using the $\lambda_{jdet}$ regularizer. 

## Table of Contents

[TOC]

## Project Goals

* **Reproduction of table (3D)**
    * Original table:

| **OASIS** dataset                        | Avg. Dice (28) $\uparrow$ | $\mathcal{D}_{\psi}(x)\leq 0$ $(r^\mathcal{D}) \downarrow$ | $\mathcal{D}_{\psi}(x)\leq 0$ $(s^\mathcal{D}) \downarrow$ |
|:---------------------------------------- | ------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| SYMNet                                   | $0.743 \pm 0.113$         | $0.026\%$                                                  | -                                                          |
| SyN                                      | $0.729 \pm 0.109$         | $0.026\%$                                                  | $0.005$                                                    |
| NiftyReg                                 | $0.775 \pm 0.087$       | $0.102\%$                                                  | $1395.988$                                                 |
| Log-Demons                               | $0.764 \pm 0.098$        | $0.121\%$                                                  | $84.904$                                                   |
| NODEO (original paper $\lambda_1 = 2.5$) | $0.778 \pm 0.026$         | $0.030\%$                                                 | $34.183$                                                   |
| NODEO (original paper $\lambda_1 = 2$)   | $0.779 \pm 0.026$         | $0.030\%$                                                 | $61.105$                                                   |



* **Reproduction of 2d transformation**

![](https://i.imgur.com/9yqhoYY.jpg)
    
* **Reproduction of deformation field**
Our goal is to reproduce table 1 and figure 4 from the aforementioned [paper](https://arxiv.org/abs/2108.03443). 
Figure 4 shows the effect of Gaussian Smoothing and the $\lambda_{jdet}$ regularizer. Visualized by warped moving images, a grid visualization of the deformation field and the Jacobian determinants of the deformation, and the regions of the deformation field with negative Jacobians. We have focused on reproducing the visualization of the warped moving image and of the grid the deformation field, using the $\lambda_{jdet}$ regularizer. 
The results in table 1 are presented in two parts, with the top part showing the average dice scores over 28 structures in the OASIS data setting, while the bottom part reports both mean dice on 28 and 32 structures in the CANDI data setting. The numbers are represented as mean or mean ± std. Our task is to reproduce the results using the OASIS data setting. 
## Methods
In order to accomplish the reproduction of figure 4, the provided code had to be adjusted so it would work for two dimensional images as input. Furthermore, we had to write a function to visualize the grid visualization of the deformation field. 

The code provided was functional for 3D image inputs, but lacked documentation and organization. To adapt it for 2D images, we used slices from the dataset as input. During debugging, we discovered that numerous functions required reshaping and adjustment. Though time-consuming, this process aided our comprehension of the code and ultimately resulted in success. The primary functions we modified were those responsible for windowing and calculating the loss.

The main goal of reproduction of table 1 is to get the results of the author's study on image registration using the standard OASIS dataset. We aim to reproduce the table generated in the study by adapting the code to work with the OASIS dataset, despite the differences in dataset structure compared to the author's specialized training dataset. Our task involves matching the labels between the OASIS dataset and the specialized dataset used by the author for image registration. For aligning the labels, we match the anatomical structures in the supplementary document of the paper with instructions of 3D OASIS dataset. Additionally, we have made modifications to the uploaded code, specifically converting the non-recursive Registration_3D function into a loop version for improved efficiency. By successfully reproducing the table and aligning the labels, we aim to validate the findings of the original study using the publicly available OASIS dataset, while focusing on the image registration aspect of the research.

In this study, we employed an atlas-based method to match pictures 2-10 using picture 1 as the atlas, and pictures 12-20 using picture 11 as the atlas, all the way up to picture 41 as the atlas for matching pictures 42-50. We calculated the Dice similarity coefficient and the number and proportion of elements in the Jacobian matrix that were less than zero, for both $\lambda=2$ and $\lambda=2.5$. Overall, there are a total of 100 registration tasks between images in this study.
## Results
* Table for training data using OASIS with the NODEO method under the conditions of lamda=2 and lamda=2.5

| **OASIS** dataset                        | Avg. Dice (28) $\uparrow$ | $\mathcal{D}_{\psi}(x)\leq 0$ $(r^\mathcal{D}) \downarrow$ | $\mathcal{D}_{\psi}(x)\leq 0$ $(s^\mathcal{D}) \downarrow$ |
|:---------------------------------------- | ------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- |
| NODEO (reproduced $\lambda_1 = 2.5$)                                   | $0.778 \pm 0.071$         | $0.067\%$                                                  | $256.73$                                                          |
| NODEO (reproduced $\lambda_1 = 2$)                                      | $0.869 \pm 0.050$         | $0.047\%$                                                  | $108.71$                                                    |

In the table below, the results of warping a 2D slice can be seen:


| Fixed image (Subject 10) | Moving image (Subject 20)| Resulting warped image |
| -------- | -------- | -------- |
| ![Fixed](https://i.imgur.com/msit2F8.png)    | ![Moving](https://i.imgur.com/9pZb7MM.png)     | ![Warped](https://i.imgur.com/LjUtM96.png) |




The resulting deformation field is shown here as well.
![](https://i.imgur.com/BnwGL9w.png)




## Discussion
Discussion of the results, is the paper reproducable?
Problems:
* Code only published for 3D, even though results for 2D transformations are included.
* Not all images used in report could be generated easily

Although the article and we both use the OASIS dataset, the range of values for the "label" variable in the program can reach 60, while in the OASIS dataset, we downloaded, each element in the "label" array has a value range of 0 to 33. This suggests that the dataset used by the author for training may have been preprocessed differently from our dataset. Due to the difference in the labels, it is foreseeable that the results obtained from training on these two datasets may differ. 

In our analysis, we observed that the average Dice coefficient value obtained using the NODEO method with the OASIS dataset was slightly different from the results reported in the original paper, particularly under the condition of "$\lambda_1= 2$". However, the mean Dice coefficient value we obtained was 0.869, which is even higher than the value reported by the authors. This suggests that the better performance of the NODEO method, as observed in our analysis, is justifiable.

Regarding the additional performance criteria, we also evaluated the total number of negative entries in the Jacobian matrix and the ratio of negative entries to the total number of entries, as indicated in the table. We found that we got a low value in these criteria by using NODEO. But it still performs better than previous methods. Specifically, the number of negative Jacobian entries of NiftyReg and Log-Demons, is 0.102% and 0.121%, respectively, both higher than the ratio we reproduced by NODEO. 



## Conclusions

For the evaluation of a model, it's important to note that different research projects or programs may use different label values depending on their specific needs or research goals. Therefore, it's always a good practice to check the data and label values before using them for any analysis or training. In this scenario, it may be necessary to establish a mapping between the label values used in the program and the label values present in the downloaded OASIS dataset, to ensure consistency in subsequent analyses or training procedures. Alternatively, one could seek clarification from the supplementary document of the paper and the instructions released by creators of the OASIS dataset regarding the differences in label values and their potential impact on results. Such steps can help to ensure methodological rigor and accuracy in data-driven research.






## Task Division


| Task     | Person      |
| -------- | -------- |
| Adjusting original code to implement support for 2D images | Givani & Thomas|
|Reproduction of Tables | Han|
|Reproduction of Deformation field| Givani & Thomas |




## Appendix

:::info
**Find this document incomplete?** Leave a comment!
:::

###### tags: `NODEO` `Reproduction` 'Deep Learning'

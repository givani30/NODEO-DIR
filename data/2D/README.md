# [Neurite](https://github.com/adalca/neurite) OASIS Sample Data

Organized data collection including 414 subjects from the 
[open-access OASIS dataset](oasis-brains.org) processed with FreeSurfer and SAMSEG.  
If you use these data, please see the [reference section](#Reference) below for citations.


[Download v1.0 here](http://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar)

# Structure

Each subdirectory contains original and normalized T1 scan data as well as
label segmentations for an individual subject. The list of subject IDs can
be found in the `subjects.txt` file.

Each subject directory has raw (`orig`) and skull-stripped / bias-corrected (`norm`)
images in the original scanner space and resampled into an affinely-aligned, common
template space. Additionally, we provide an aligned coronal-slice for 2D applications.
In summary, each subject directory contains the following images:

    FILENAME              SHAPE             SPACE

    orig.nii.gz           256 x 256 x 256   raw image in scanner space
    norm.nii.gz           256 x 256 x 256   corrected image in scanner space

    aligned_orig.nii.gz   160 x 192 x 224   raw image aligned in template space
    aligned_norm.nii.gz   160 x 192 x 224   corrected image aligned in template space

    slice_orig.nii.gz     160 x 192         2D raw image aligned in template space
    slice_norm.nii.gz     160 x 192         2D corrected image aligned in template space

We conformed all images to a common shape and scaled them between 0 and 1, and skullstripped and
bias-corrected 'norm' images with freesufer. We 
registered and resampled the images into freesurfer's talairach space using the talairach.xfm atlas
transform generated by recon-all and cropped via `[(48, 48), (31, 33), (3, 29)]`. We also extracted 
coronal slice `109` from these cropped volumes to generate the 2D slice images.

Each subject contains a set of corresponding automated label segmentations. 
We provide a 35-label segmentation of major anatomical regions as well as a 4-label tissue-type 
segmentation, generated from the former. For the 2D images, we instead provide a 24-label 
segmentation, consisting of the most common structures found in that sliced region. 
Corresponding mappings from label ID to structure name are available in the following files:

    seg35_labels.txt
    seg24_labels.txt
    seg4_labels.txt

These are provided in freesurfer's colortable file format, which can be used to
visualize the segmentations correctly in [freeview](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeviewGuide/FreeviewIntroduction).
For example:

    cd OASIS_OAS1_0445_MR1
    freeview norm.nii.gz \
             seg4.nii.gz:colormap=lut:lut=../seg4_labels.txt \
             seg35.nii.gz:colormap=lut:lut=../seg35_labels.txt


# Reference

These data were prepared by Andrew Hoopes and Adrian V. Dalca for the following 
HyperMorph paper. If you use this collection please cite the following and refer to the 
[OASIS Data Use Agreement](oasis-brains.org/#access).

    HyperMorph: Amortized Hyperparameter Learning for Image Registration.
    Hoopes A, Hoffmann M, Fischl B, Guttag J, Dalca AV. 
    arXiv preprint arXiv:2101.01035, 2021. https://arxiv.org/abs/2101.01035

    Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in
    Young, Middle Aged, Nondemented, and Demented Older Adults.
    Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL.
    Journal of Cognitive Neuroscience, 19, 1498-1507.
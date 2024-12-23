# Brain Age Estimation

In this project, I tried to build on the work covered in the previous literature of Brain Age Estimation. I used the same dataset as used in my first brain age estimation repository: https://github.com/sohini-02/re-bae_resnet10.git. Here, instead of using 1 middle axial slice from each patient, I used three middle slices per patient: the sagittal, coronal and axial slices from the T1 weighted data available in the original dataset. 

These three slices were processed together as an order-invariant set using the DeepSets module, in the hopes that the model would be able to learn the intricacies of relation between the images in each set, thus giving it more predictive power. However, it was seen that the model was overfitting spectacularly, which is not great news. I'm currently still working on this project to try and make it stop overfitting because I still believe it possible to achieve a higher accuracy than before using this approach.

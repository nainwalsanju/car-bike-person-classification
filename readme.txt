GRAZ-02 database
============================================================

author: Andreas Opelt and Axel Pinz
email: opelt-removethisperhand- <aT> emt.tugraz.at or axel.pinz@tugraz.at
date: 13.07.2004
last update: 26.11.2004 (added ground truth files) 
institution:  Institute of Electrical Measurement and Measurement Signal Processing
              Graz, University of Technology, Austria 
              
============================================================

IMPORTANT NOTE (2007-06-05):
----------------------------
This database has been modified by Marcin Marszalek and Cordelia Schmid. 
Ground truth annotation has been improved significantly. 
See their CVPR 2007 paper:
Marcin Marszalek and Cordelia Schmid. Accurate Object Localization with
Shape Masks. IEEE Conference on Computer Vision & Pattern Recognition, 2007.

Their improved version of the database "IG02" - INRIA annotations for GRAZ-02
is available at
http://lear.inrialpes.fr/people/marszalek/data/ig02/

============================================================

CONTENT:
--------

Database for object recognition or object categorization.
Containing images with objects of high complexety,
a high intra-class variability on highly cluttered backgrounds.

3 categories (bikes, persons, cars) and one counter-class (bg_graz).
Contains 365 images with bikes, 311 images with persons, 420 images with cars 
and 380 images not containing one of these objects. 

Also available here the ground truth for 300 images of each category.
It is given in terms of pixel segmentation masks with values between 0 and 255.
Where pixels with 0 denote the object in the image.


Difference to GRAZ-01:
----------------------
More complexity, a third category and a better balanced appearance 
of the backgrounds of the various classes.

============================================================


Related Publications:
---------------------

Generic object recognition with boosting.	
A. Opelt, M. Fussenegger, A. Pinz and P. Auer.  
Technical Report TR-EMT-2004-01, EMT, TU Graz, Austria, 2004. 
Submitted to the IEEE Transactions on Pattern Analysis and Machine Intelligence. 
Download available in this directory.

A. Opelt and A. Pinz. Object localization in weakly supervised object recognition. 
In REview for the IEEE Conference on Computer Vision and Pattern Recognition, San Diego, California, June, 2005. 
Will soon be available for download.


============================================================

Acknowledgements:
-----------------

The EU project LAVA (IST-2001-34405) and the Austrian Science Foundation (project S9103-N04)

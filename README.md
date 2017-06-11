**Machine Learning: Facial-And-Gender-Recognition**

This project is used to recognize faces and genders from a subset of the FaceScrub dataset.  
The dataset consists of URLs of images with faces, as well as the bounding boxes of the faces.

The image format is x1,y1,x2,y2, where (x1,y1) is the coordinate of the top-left corner of the 
bounding box and (x2,y2) is that of the bottom-right corner, with (0,0) as the top-left corner of the image.
Assuming the image is represented as a Python NumPy array I, a face in I can be obtained as I[y1:y2, x1:x2].

# NvidiaWarpImageProcessing

## Description
Project implements parrelizable image processing using Nvidia Warp. Implements both noise reduction and sharpening algorithms. 
Can be ran serially on the CPU, or in parallel if ran with an Nvidia GPU. 

Image sharpening is done through an unsharp mask based algorithm. First the image is blurred using a simple mean averaging filter for noise removal. 
Then the blurred image is subtracted from the original one, creating an edge image. This image highlights high intensity areas, and extracts important features. 
This edge image is then added to the original with a multiplier, applying a variable degree of sharpening to the output image.

Noise reduction is based on a gaussian filter. Pixels within the kernel closer to the center are given a higher weight, which is reduced
when spread further out. The distribution of the filter can be changed with a larger value of the passed paramater, causing a wider spread, which 
leads to increased blurring. As this factor is increased, the dimensions of the kernel should as well.

Border handling is done through reflection. Any points when calculating the image kernels that fall outside of image boundaries are
reflected, choosing a value that exists in the image.


## Getting Started

### Dependencies
Python libraries numpy, pillow and warp-lang
```pip install numpy pillow warp-lang```

### Executing program
Change cpu -> cuda if computer is equiped with Nvidia graphics card (a3.py line 7)

* How to build and run the program
python3 a3.py  algType(-s (sharpen) -n (noise removal)) kernSize param inFileName (image in) outFileName (image out)

-s Applies sharpening algorithm to the image. Uses unsharp masking
    param - scaling constant for the impact of the edge image. Recommended values of 0.3-0.7 

-n Applies noise reduction algorithm to the image. Uses gaussian filtering
    param - used for gaussian weights, is the scale of weight distribution of the neighbourhood. Larger values increase blurring. Increase kernel dimensions in correlation to weight increase

kernsize 3 (3x3 grid), 5 (5x5 grid) etc

eg: python3 a3.py -s 3 0.3 test.jpg out.jpg

## Author Information
Lucas Sudol
Sebastian Kula


## Acknowledgments

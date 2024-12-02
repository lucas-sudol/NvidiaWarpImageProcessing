# NvidiaWarpImageProcessing

## Description
Project implements parrelizable image processing using Nvidia Warp. Implements both noise reduction and sharpening algorithms. Can be ran serially on the CPU, or in parallel if ran with an Nvidia GPU. 

Image sharpening is done through a unsharp mask mased algorithm. First the image is blurred using a simple mean averaging filter for noise removal. Then the blurred image is subtracted from the original one, created an edge image. This image highlights high intensity areas, and extracts important features. 

## Getting Started

### Dependencies
Python libraries numpy, pillow and warp-lang
```pip install numpy pillow warp-lang```

### Executing program
Change "cpu" to "cuda" if your computer is equipped with a Nvidia graphics card (a3.py line 7)

* How to run the program: 

    ```python3 a3.py <algType> <kernSize> <param> <inFileName> <outFileName>```

Arguments:
```
    <algType>       Algorithm type: specify "-s" for sharpening or "-n" for noise removal
    <kernSize>      Size of the kernel (e.g., 3 (3x3 grid), 5 (5x5 grid), 7 (7x7 grid))
    <param>         Parameter for the algorithm (e.g., intensity factor)
    <inFileName>    Input image file name (e.g., input.jpg)
    <outFileName>   Output image file name (e.g., output.jpg)
```

-s Applies sharpening algorithm to the image. Uses unsharp masking
    param - scaling constant for the impact of the edge image. Recommended values of 0.3-0.7 

-n Applies noise reduction algorithm to the image. Uses gaussian filtering
    param - used for gaussian weights, is the scale of weight distribution of the neighbourhood. Larger values increase blurring. Increase kernel dimensions in correlation to weight increase

Example:
```
  python3 a3.py -s 3 0.3 input.jpg output.jpg
```

## Author Information
Lucas Sudol
Sebastian Kula


## Acknowledgments
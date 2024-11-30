import sys
import numpy as np
from PIL import Image

def sharpen_image(numpyArr, kernSize, param):
    # sharpening kernel
    pass

def noise_removal(numpyArr, kernSize, param):
    # noise removal kernel
    pass

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 a3.py algType kernSize param inFileName outFileName")
        sys.exit(1)

    algType = sys.argv[1]
    kernSize = int(sys.argv[2])
    param = float(sys.argv[3])
    inFileName = sys.argv[4]
    outFileName = sys.argv[5]

    # Ensure kernSize is positive and odd
    if kernSize <= 0 or kernSize % 2 == 0:
        print("Error: kernSize must be a positive odd number.")
        sys.exit(1)

    # Load the image
    image = Image.open(inFileName)
    numpyArr = np.asarray(image, dtype='float32')

    # Apply the selected algorithm
    if algType == "-s":
        result = sharpen_image(numpyArr, kernSize, param)
    elif algType == "-n":
        result = noise_removal(numpyArr, kernSize, param)
    else:
        print("Error: algType must be '-s' for sharpen or '-n' for noise removal.")
        sys.exit(1)

    # Save the processed image
    result_image = Image.fromarray(result.astype('uint8'))
    result_image.save(outFileName)

if __name__ == "__main__":
    main()

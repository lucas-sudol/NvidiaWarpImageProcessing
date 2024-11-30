import sys
import numpy as np
from PIL import Image
import warp as wp

wp.init()
computeDevice = "cpu" # "cuda" for gpu

@wp.kernel
def sharpen_kernel(input_image: wp.array(dtype=float), output_image: wp.array(dtype=float), kern_size: int, param: float):
    # sharpening kernel
    pass

@wp.kernel
def noise_removal_kernel(input_image: wp.array(dtype=float), output_image: wp.array(dtype=float), kern_size: int, param: float):
    # noise removal kernel
    pass

def apply_kernel(kernel, input_array, kern_size, param):
    output_array = np.zeros_like(input_array)

    input_gpu = wp.array(input_array, dtype=wp.float32, device=computeDevice)
    output_gpu = wp.array(output_array, dtype=wp.float32, device=computeDevice)

    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=input_gpu.shape[:2],  # Assuming 2D image
        inputs=[input_gpu, output_gpu, kern_size, param]
    )

    # Copy the result back to CPU
    return output_gpu.numpy()

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 a3.py algType kernSize param inFileName outFileName")
        sys.exit(1)

    algType = sys.argv[1]
    kernSize = int(sys.argv[2])
    param = float(sys.argv[3])
    inFileName = sys.argv[4]
    outFileName = sys.argv[5]

    # Check kernSize
    if kernSize <= 0 or kernSize % 2 == 0:
        print("Error: kernSize must be a positive odd number.")
        sys.exit(1)

    # Load image and convert to numpy array
    image = Image.open(inFileName)
    numpyArr = np.asarray(image, dtype='float32')

    # Choose algorithm based on user input
    if algType == "-s":
        result = apply_kernel(sharpen_kernel, numpyArr, kernSize, param)
    elif algType == "-n":
        result = apply_kernel(noise_removal_kernel, numpyArr, kernSize, param)
    else:
        print("Error: algType must be '-s' for sharpen or '-n' for noise removal.")
        sys.exit(1)

    # Save the output image
    result_image = Image.fromarray(result.astype('uint8'))
    result_image.save(outFileName)

if __name__ == "__main__":
    main()

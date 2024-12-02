import sys
import numpy as np
from PIL import Image
import warp as wp

wp.init()
computeDevice = "cuda" # "cuda" for gpu

@wp.kernel
def sharpen_kernel_1(input_image: wp.array(dtype=wp.float32, ndim=2), output_image: wp.array(dtype=wp.float32, ndim=2), kern_size: int, param: float):
    # sharpening kernel
    i, j = wp.tid()
    flipx = 1
    flipy = 1

    #Get offsets for i,j for 
    for x in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
        for y in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
            #Border Handling by reflection
            if i+x < 0 or i+x > output_image.shape[0]: #check if tile is below 0 or above max x value
                flipx = -1

            if j+y < 0 or j+y > output_image.shape[1]: #check if tile is below 0 or above max y value
                flipy = -1

            #Accumulate tiles to output image
            output_image[i, j] += input_image[i + x*flipx, j + y*flipy]
            flipx = 1
            flipy = 1

    #Divide by number of tiles to find mean
    output_image[i, j] *= (1.0/float(kern_size * kern_size))
    output_image[i, j] = input_image[i,j] + param*(input_image[i,j] - output_image[i, j])

    if(output_image[i, j] > 255.0):
        output_image[i, j] = 255.0

    pass

@wp.kernel
def sharpen_kernel_3(input_image: wp.array(dtype=wp.float32, ndim=3), output_image: wp.array(dtype=wp.float32, ndim=3), kern_size: int, param: float):
    # sharpening kernel
    i, j, k = wp.tid()
    flipx = 1
    flipy = 1

    #Get offsets for i,j for 
    for x in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
        for y in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
            #Border Handling by reflection
            if i+x < 0 or i+x > output_image.shape[0]: #check if tile is below 0 or above max x value
                flipx = -1

            if j+y < 0 or j+y > output_image.shape[1]: #check if tile is below 0 or above max y value
                flipy = -1

            #Accumulate tiles to output image
            output_image[i, j, k] += input_image[i + x*flipx, j + y*flipy, k]
            flipx = 1
            flipy = 1

    #Divide by number of tiles to find mean
    output_image[i, j, k] *= (1.0/float(kern_size * kern_size))
    output_image[i, j, k] = (input_image[i,j,k] + param*(input_image[i,j,k] - output_image[i,j,k]))

    if(output_image[i, j, k] > 255.0):
        output_image[i, j, k] = 255.0

    pass

@wp.kernel
def noise_removal_kernel_1(input_image: wp.array(dtype=wp.float32, ndim=2), output_image: wp.array(dtype=wp.float32, ndim=2), kern_size: int, param: float):
    # noise removal kernel
    i, j = wp.tid()
    flipx = 1
    flipy = 1

    weights = float(0.0)
    weight = float(0.0)


    #Get offsets for i,j for 
    for x in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
        for y in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
            #Border Handling by reflection
            if i+x < 0 or i+x > output_image.shape[0]: #check if tile is below 0 or above max x value
                flipx = -1

            if j+y < 0 or j+y > output_image.shape[1]: #check if tile is below 0 or above max y value
                flipy = -1

            weight = wp.exp(-(float(x) * float(x) + float(y) * float(y)) / (2.0 * param * param))
            weights += weight

            #Accumulate tiles to output image
            output_image[i, j] += (input_image[i + x*flipx, j + y*flipy] * weight)
            flipx = 1
            flipy = 1

    #Divide by number of tiles
    output_image[i, j] *= (1.0 / weights)
    pass

@wp.kernel
def noise_removal_kernel_3(input_image: wp.array(dtype=wp.float32, ndim=3), output_image: wp.array(dtype=wp.float32, ndim=3), kern_size: int, param: float):
    # noise removal kernel
    i, j, k = wp.tid()
    flipx = 1
    flipy = 1

    weights = float(0.0)
    weight = float(0.0)


    #Get offsets for i,j for 
    for x in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
        for y in range(-(kern_size-1)/2, (kern_size-1)/2 + 1):
            #Border Handling by reflection
            if i+x < 0 or i+x > output_image.shape[0]: #check if tile is below 0 or above max x value
                flipx = -1

            if j+y < 0 or j+y > output_image.shape[1]: #check if tile is below 0 or above max y value
                flipy = -1

            weight = wp.exp(-(float(x) * float(x) + float(y) * float(y)) / (2.0 * param * param))
            weights += weight

            #Accumulate tiles to output image
            output_image[i, j, k] += (input_image[i + x*flipx, j + y*flipy, k] * weight)
            flipx = 1
            flipy = 1

    #Divide by number of tiles
    output_image[i, j, k] *= (1.0 / weights)
    pass










#Initialize Kernel
def apply_kernel(kernel, input_array, kern_size, param, channels):
    output_array = np.zeros_like(input_array)

    inputWp = wp.array(input_array, dtype=wp.float32, device=computeDevice)
    outputWp = wp.array(output_array, dtype=wp.float32, device=computeDevice)

    # Launch kernel
    wp.launch(
        kernel=kernel,
        dim=inputWp.shape[:channels],
        inputs=[inputWp, outputWp, kern_size, param],
        device=computeDevice
    )

    # Copy the result back to CPU
    return outputWp.numpy()

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 a3.py  algType (-s (sharpen) -n (noise removal)) kernSize param inFileName outFileName")
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
    
    mode = image.mode
    if mode == 'L':
        channels = 2
    elif mode == 'RGB' or mode == 'RGBA':
        channels = 3
    else:
        print("Error: unsupported image mode.")
        sys.exit(1)
    
    numpyArr = np.asarray(image, dtype='float32')

    # Choose algorithm based on user input
    if algType == "-s": #Sharpen
        if channels == 2:
            result = apply_kernel(sharpen_kernel_1, numpyArr, kernSize, param, channels)
        else:
            result = apply_kernel(sharpen_kernel_3, numpyArr, kernSize, param, channels)

    elif algType == "-n": #De Noise
        if channels == 2:
            result = apply_kernel(noise_removal_kernel_1, numpyArr, kernSize, param, channels)
        else:
            result = apply_kernel(noise_removal_kernel_3, numpyArr, kernSize, param, channels)
            

    else:
        print("Error: algType must be '-s' for sharpen or '-n' for noise removal.")
        sys.exit(1)

    print(channels)

    # Save the output image
    result_image = Image.fromarray(result.astype('uint8'))
    result_image.save(outFileName)

if __name__ == "__main__":
    main()

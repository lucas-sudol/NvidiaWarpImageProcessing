import sys
import numpy as np
from PIL import Image

image = Image.open(sys.argv[1])

# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)

numpyArr = np.asarray(image, dtype='float32')
print(numpyArr.shape)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# this class takes in an image containing words as input and segments the words into individual characters represented as arrays
class Segmenter:

    # loads in image and creates arrays with number of vertical/horizontal pixels in each column/row
    def __init__(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=1)
        self.image = dilated

        self.horizontal = np.sum(self.image,axis=1,keepdims=True)/255
        self.flattened = np.ndarray.flatten(self.image)

    # combines all segmenting methods
    def segment(self):
        self.lineLevelSegment()
        self.wordSegment()
        self.characterSegment()

    # breaks up the original image into new images corresponding to each line of handwriting
    def lineLevelSegment(self):
        # finds the indices to split the row
        indices = np.array(np.where(self.horizontal < 1))
        indices = indices.reshape((indices.size))
        indices = indices[indices != 0]
        self.lines = []
        if (len(indices) > 0):
            indices = np.array(median_sequences(indices))
            splitImage = []

            # appends the new rows to new array
            for i in range(len(indices)-1):
                start = indices[i]
                end = indices[i+1]
                sliced_part = self.image[start:end, :]
                splitImage.append(sliced_part)

            self.lines = splitImage
 
    def wordSegment(self):
        self.words = []
        # vertical projection for each line
        # get indices where there are no pixels along that column
        for line in self.lines:
            vertical_projection = np.sum(line, axis = 0)/255
            index = np.array(np.where(vertical_projection < 1))
            index = index.reshape((index.size))
            
            # count the number of consecutive columns with no pixels
            counts = []
            count = 0
            for i in range(len(index)-1):
                if (index[i] - index[i+1] == -1):
                    count = count + 1
                else:
                    counts.append(count)
                    count = 0
            counts.append(count)

            # differentiate between word gaps and character gaps
            index = median_sequences(index)
            
            # calculate threshold value for word gaps vs. character gaps
            mean_gap = sum(counts)/len(counts)
            std_dev_gap = (sum((gap - mean_gap)**2 for gap in counts) / len(counts))**0.5
            threshold = mean_gap + 0.75 * std_dev_gap

            word_split_index = []

            # find indices where words are split
            word_split_index.append(math.floor(2 * index[0] - threshold/2))
            for i in range(1, len(index)-1):
                if (counts[i] > threshold):
                    word_split_index.append(index[i])
            word_split_index.append(math.ceil(line.shape[1] - 2 * (line.shape[1] - index[len(index)-1]) + threshold/4))

            # use indices to get words
            for i in range(len(word_split_index)-1):
                start = word_split_index[i]
                end = word_split_index[i+1]
                sliced_part = line[:, start:end]
                self.words.append(sliced_part)

    # use the same process as other segmentation to segment the characters
    def characterSegment(self):
        self.characters = []
        for word in self.words:
            
            # set up indices
            vertical_projection = np.sum(word, axis = 0)/255
            index = np.array(np.where(vertical_projection < 1))
            index = index.reshape((index.size))
            if (len(index) > 0):
                index = median_sequences(index)

            # split into characters and add to new array
            for i in range(len(index)-1):
                start = index[i]
                end = index[i+1]
                sliced_part = word[:, start:end]
                self.characters.append(sliced_part)

            self.characters.append([])

# get the median of each consecutive sequence of integers in an array
# ex. passing [1,2,3, 5,6,7, 10,11,12] returns [2,6,11]
def median_sequences(arr):
    result = []
    start = 0
    end = 0
    for x in range(len(arr) - 1):
        if (arr[x+1] - arr[x] != 1):
            end = x
            count = 0

            result.append(math.floor((arr[start]+arr[end])/2))
            start = x+1

    end = len(arr)-1
    result.append(math.floor((arr[start]+arr[end])/2))

    return result

# finds second biggest number of an array
def find_second_biggest(numbers):
    copyN = numbers.copy()
    maxCopy = max(copyN)
    copyN.remove(maxCopy)
    second_biggest = max(copyN)
    return second_biggest






        
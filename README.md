# BFS Edge Detection in Images

- This project uses Breadth-First Search (BFS) to detect edges in grayscale images based on a specified threshold and starting coordinates. The main functions provided are `generate_image` and `generate_image_grid`.
- Team members : Chirag Sehgal , Akshat Patiyal , Jasjyot Gulati
## Requirements

Ensure you have the following libraries installed:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

You can install them using pip if they are not already installed:

```sh
pip install opencv-python-headless numpy matplotlib
```

## Usage

To generate a single edge-detected image, use the `generate_image` function. This function uses a predefined starting coordinate and threshold value.

```python
if __name__ == "__main__":
    # Example usage
    image_path = "./testImages/lotus.png"
    generate_image(image_path)
```
To generate a grid of edge-detected images with various starting coordinates and threshold values, use the generate_image_grid function.
```python
if __name__ == "__main__":
    # Example usage
    image_path = "./testImages/lotus.png"
    generate_image_grid(image_path)
```

The generate_image_grid function will produce a grid of images with starting coordinates ranging from (0, 0) to (500, 500) with steps of 50, and threshold values ranging from 20 to 100 with steps of 20.

### To Find coordinates in the image
Change `image_path` in `mouseHover.py` and run it. On hovering your mouse over the image, the coordinates of your mouse will appear in the output window.


## Functions

### `bfs(image, start, threshold)`

Performs BFS on the grayscale image from the `start` coordinate, marking pixels as edges if the difference in pixel intensity exceeds the `threshold`.

**Parameters:**
- `image`: The grayscale image as a NumPy array.
- `start`: A tuple `(x, y)` representing the starting coordinate for BFS.
- `threshold`: An integer threshold for detecting edges.

**Returns:**
- `edges`: A boolean NumPy array where edges are marked as `True`.

### `main(image_path, start_coord, threshold)`

Reads the image, converts it to grayscale, and uses `bfs` to find edges. Returns an image with edges highlighted.

**Parameters:**
- `image_path`: Path to the image file.
- `start_coord`: Starting coordinate for BFS.
- `threshold`: Threshold for detecting edges.

**Returns:**
- `resized_image`: The resulting image with edges highlighted, resized to 500x500 pixels.

### `generate_image_grid(image_path)`

Generates a grid of edge-detected images using a combination of different starting coordinates and thresholds. Displays the images in a grid format.

**Parameters:**
- `image_path`: Path to the image file.

### `generate_image(image_path)`

Generates a single edge-detected image based on specified starting coordinates and threshold. Displays the original and processed images side by side.

**Parameters:**
- `image_path`: Path to the image file.

## Explanation

1. **BFS Edge Detection**:
   - The BFS algorithm starts from a given pixel and explores its 4-connected neighbors (up, down, left, right).
   - If the difference in pixel intensity between the current pixel and the starting pixel is within the threshold, the neighbor is added to the queue for further exploration.
   - If the difference exceeds the threshold, the neighbor is marked as an edge.

2. **Image Processing**:
   - The original image is read and converted to grayscale.
   - BFS is applied to detect edges, and the resulting edges are highlighted in the output image.
   - The output image is resized to 500x500 pixels for uniformity.

3. **Visualization**:
   - The `generate_image` function displays the original and edge-detected images side by side for comparison.
   - The `generate_image_grid` function generates a grid of images with varying parameters, providing a comprehensive view of the edge detection results.

By following the usage instructions and understanding the explanation, you can easily apply BFS-based edge detection to your images and visualize the results in various ways.

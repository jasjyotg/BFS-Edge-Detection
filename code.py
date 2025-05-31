import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

def bfs(image, start, threshold):
    rows, cols = image.shape
    visited = np.zeros((rows, cols), dtype=bool)
    edges = np.zeros((rows, cols), dtype=bool)
    
    # Directions for the 4-connected neighbors (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Queue for BFS
    queue = deque([start])
    visited[start] = True
    start_value = image[start]
    
    while queue:
        x, y = queue.popleft()
        
        for direction in directions:
            nx, ny = x + direction[0], y + direction[1]
            
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                if abs(int(image[nx, ny]) - int(start_value)) <= threshold:
                    visited[nx, ny] = True
                    queue.append((nx, ny))
                else:
                    edges[nx, ny] = True

    return edges

def main(image_path, start_coord, threshold):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #apply Gaussian Blur
    grayscale_image = cv2.GaussianBlur(grayscale_image,(3,3),0)

    # Get the image dimensions
    rows, cols = grayscale_image.shape
    
    # Check if the start_coord is within bounds
    if not (0 <= start_coord[0] < rows and 0 <= start_coord[1] < cols):
        print(f"Warning: Coordinates {start_coord} are out of bounds for the image. Skipping this coordinate.")
        return None
    
    # Perform BFS to find the edges of the object
    edges = bfs(grayscale_image, start_coord, threshold)
    
    # Create an empty black image to only display edges
    edge_only_image = np.zeros_like(image)
    
    # Convert edges to a red line (BGR format)
    edge_only_image[edges] = [255, 255, 255]  # Set edge pixels to red
    
    # Optional: Increase the visibility of edges by making them thicker
    kernel = np.ones((3, 3), np.uint8)  # A 3x3 kernel for dilation
    dilated_edges = cv2.dilate(edge_only_image, kernel, iterations=1)
    
    # Resize the output image to 500x500
    resized_image = cv2.resize(dilated_edges, (500, 500))
    
    return resized_image  # Return the resized image

def generate_image_grid(image_path, output_size=(2000, 1500), dpi=300):
    images = []
    start_coords = []
    thresholds = []
    
    for j in range(0, 501, 50):
        for i in range(20, 101, 20):
            # Get the processed image for each combination of start_coord and threshold
            img = main(image_path, start_coord=(j, j), threshold=i)
            if img is not None:
                images.append(img)
                start_coords.append((j, j))
                thresholds.append(i)
    
    if not images:
        print("No valid images were generated due to coordinate issues.")
        return
    
    # Plotting the images in a grid using matplotlib
    fig, axes = plt.subplots(5, 10, figsize=(20, 15))  # Adjusted figure size for clarity
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjusted spacing between subplots

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):  # Avoid IndexError if the grid size is larger than the number of images
            ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB), aspect='auto')
            ax.axis('off')  # Hide axes for a cleaner image
            # Label the axes with the start coordinate and threshold
            ax.set_title(f"Coord: {start_coords[idx]} | T: {thresholds[idx]}", fontsize=8, pad=-1)
        else:
            ax.axis('off')  # Hide axes for empty subplots
    
    plt.tight_layout()

    output_dir = "./rendersGaussian/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename and save
    image_filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, image_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Image grid saved to {save_path}")
    
    plt.show()

def generate_image(image_path):
    # Get the processed image
    start_coord=(250, 250)
    threshold=40
    img = main(image_path, start_coord, threshold)
    
    # Read the original image
    original_img = cv2.imread(image_path)
    
    # Convert images to RGB for proper display in matplotlib
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a subplot with 1 row and 2 columns, reduce figsize to make images slightly smaller
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Adjusted figure size for smaller images
    
    # Display the original image on the left
    axes[0].imshow(original_img_rgb)
    axes[0].set_title("Original Image", pad=0,fontsize = 20)  # Adjusted padding to move title lower
    axes[0].axis('off')  # Hide axes for cleaner display
    
    # Display the BFS edge detection result on the right
    axes[1].imshow(img_rgb)
    axes[1].set_title(f"BFS Edge Detection \nCoord: {start_coord} | Threshold: {threshold}", pad=0,fontsize = 20)  # Adjusted padding to move title lower
    axes[1].axis('off')  # Hide axes for cleaner display
    
    plt.tight_layout()  # Adjust layout for spacing
    plt.show()


if __name__ == "__main__":
    # Example usage
    image = "gaussian.png"
    image_path = "./testImages/" + image
   # generate_image_grid(image_path)

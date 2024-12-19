import os
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

if __name__ == '__main__':

    print(f"Start...")

    image_url = os.path.join('images', 'eu_raw.dng')
    image = ski.io.imread(image_url)
    image = image.astype(np.float64) / 255 # Normalizar imagem
  
    # Computar SVD da imagem
    k = int(min([image.shape[0], image.shape[1]]) / 2)
    compressed_image = []
    for i in range(3):
        U, S, Vh = np.linalg.svd(image[:,:,i], full_matrices=True)
        compressed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vh[:k, :]))
        compressed_image.append(compressed)

    compressed_image = np.stack(compressed_image, axis=-1)

    image = image * 255 
    image = image.astype(np.uint16)

    compressed_image = compressed_image * 255
    compressed_image = compressed_image.astype(np.uint16)

    metric = calculate_mse(image, compressed_image)
    print(f"\nMSE : {metric}")
    print(f"Original : {image.nbytes} bytes")
    print(f"Comprimida : {compressed_image.nbytes} bytes")
    print(f"k : {k}")
    
    # Salvar a imagem comprimida
    ski.io.imsave('compressed_image.jpeg', compressed_image, quality=100)

    # Visualizar
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Imagem Comprimida")
    plt.imshow(compressed_image)
    plt.show()

import cv2
import numpy as np 
from matematica import HOSVD, reconstruct, norma_frobenius, tucker_error

def compress_image(path, compressionFactor = 0.5, blockSize = 16, progressCallback = None):
    #Citire imagine
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Nu s-a putut citi imaginea, verificati calea")
    
    if len(img.shape) == 2:
        imgRgb = np.stack([img]*3, axis=-1)
    else:
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Normalizam tensorul
    tensor = imgRgb.astype(float) / 255.0
    H, W, C = tensor.shape

    #Organizare imagine pe blocuri de 16x16
    HNew = (H // blockSize) * blockSize
    WNew = (W // blockSize) * blockSize

    blocksH = HNew // blockSize
    blocksW = WNew // blockSize

    tensor = tensor[:HNew, :WNew, :]

    #Initializare tensor final si ranguri
    tensorFinal = np.zeros_like(tensor)
    R_b = max(1, int(blockSize * compressionFactor))
    R_c = 3


    #Initializare liste pentru erori si vectori
    blockErrors = []
    normErrors = []
    G_list = []
    Us_list = []

    totalBlocks = blocksH * blocksW
    completedBlocks = 0

    for i in range(blocksH):
        for j in range(blocksW):
            rowStart = i * blockSize
            rowEnd = rowStart + blockSize
            colStart = j* blockSize
            colEnd = colStart + blockSize

            block = tensor[rowStart:rowEnd, colStart:colEnd, :]

            G, Us = HOSVD(block, [R_b, R_b, R_c])
            G_list.append(G)
            Us_list.append(Us)

            blockReconstructed = reconstruct(G, Us)
            tensorFinal[rowStart:rowEnd, colStart:colEnd, :] = blockReconstructed

            absoluteError = tucker_error(block, G, Us)
            originalNorm = norma_frobenius(block)

            blockErrors.append(absoluteError ** 2)
            normErrors.append(originalNorm ** 2)

            completedBlocks += 1
            if progressCallback is not None:
                progressCallback(completedBlocks / totalBlocks)

    imgCompressed = np.clip(tensorFinal * 255.0, 0, 255).astype(np.uint8)
    
    dimBlockCompressed = G.size + sum(U.size for U in Us)
    dimCompressedTotal = (blocksH * blocksW) * dimBlockCompressed
    dimOriginalTotal = HNew * WNew * C
    compressionRate = dimOriginalTotal / dimCompressedTotal

    absoluteError = np.sqrt(sum(blockErrors))
    originalNorm = np.sqrt(sum(normErrors))
    relativeError = (absoluteError / originalNorm) * 100
    
    return imgCompressed, compressionRate, relativeError, absoluteError, G_list, Us_list



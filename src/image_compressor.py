import cv2
import numpy as np 
from matematica import HOSVD, reconstruct, norma_frobenius, tucker_error

def compress_image(path, compressionFactor = 0.5, blockSize = 16):
    #Citire imagine
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Nu s-a putut citi imaginea, verificati calea")
    
    if len(img.shape) == 2:
        imgRgb = np.stack([img]*3, axis=-1)
    else:
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = imgRgb.astype(float) / 255.0
    H, W, C = tensor.shape

    #Organizare imagine pe blocuri de 16x16
    HNew = (H // blockSize) * blockSize
    WNew = (W // blockSize) * blockSize
    tensor = tensor[:HNew, :WNew, :]

    tensorFinal = np.zeros_like(tensor)
    R_B = max(1, int(blockSize * compressionFactor))
    R_C = 3

    blocksH = HNew // blockSize
    blocksW = WNew // blockSize

    blockErrors = []
    normErrors = []

    for i in range(blocksH):
        for j in range(blocksW):
            rStart = i * blockSize
            rEnd = rStart + blockSize
            cStart = j* blockSize
            cEnd = cStart + blockSize

            block = tensor[rStart:rEnd, cStart:cEnd, :]

            G, Us = HOSVD(block, [R_B, R_B, R_C])

            blockReconstructed = reconstruct(G, Us)
            tensorFinal[rStart:rEnd, cStart:cEnd, :] = blockReconstructed

            absoluteError = tucker_error(block, G, Us)
            originalNorm = norma_frobenius(block)

            blockErrors.append(absoluteError ** 2)
            normErrors.append(originalNorm ** 2)

    imgCompressed = np.clip(tensorFinal * 255.0, 0, 255).astype(np.uint8)
    
    dimBlockCompressed = G.size + sum(U.size for U in Us)
    dimCompressedTotal = (blocksH * blocksW) * dimBlockCompressed
    dimOriginalTotal = HNew * WNew * C
    compressionRate = dimOriginalTotal / dimCompressedTotal

    absoluteError = np.sqrt(sum(blockErrors))
    originalNorm = np.sqrt(sum(normErrors))
    relativeError = (absoluteError / originalNorm) * 100
    
    return imgCompressed, compressionRate, relativeError, absoluteError



#
# DECODER IMAGE FOR INICTEL PROJECT
#
# By: Leonardo Montoya Obeso
# Last update: april 2023

import os
import sys
import pickle

import numpy as np
import pywt
from PIL import Image
import matplotlib.colors
import matplotlib.pyplot as plt

def FindFiles():
    picklefilenames = []
    # return all files as a list
    for file in os.listdir(os.getcwd()):
        # check the files which are end with specific extension
        if file.endswith(".pkl"):
            # print path name of selected files
            picklefilenames.append(file)
    return picklefilenames

def recoverPickleData(filename):

    file = open(filename, 'rb')
    HuffmanTree = pickle.load(file)
    HuffmanCodedBand = pickle.load(file)
    HuffmanMinValue = pickle.load(file)
    minBand = pickle.load(file)
    scaleFactor = pickle.load(file)
    Qshapess = pickle.load(file)
    shapess = pickle.load(file)

    return HuffmanTree, HuffmanCodedBand, HuffmanMinValue, minBand, scaleFactor, Qshapess, shapess

def Inverse_ISRT_basedHuffman(HuffmanTree, band_binary, minval, shapes):

    # Huffman Tree to [[Symbol_i, HuffmanCode_i]]
    def HuffmanTree_to_HuffmanDictionary(tree, binary_acumulator):
        symbol_binary = []

        def Search(data, binary_acumulator):
            index_aux = 0
            while index_aux < 2:
                if isinstance(data[index_aux], tuple):
                    Search(data[index_aux], binary_acumulator + str(index_aux ^ 1))
                else:
                    symbol_binary.append(*[(data[index_aux], binary_acumulator + str(index_aux ^ 1),)])
                index_aux += 1

        Search(tree, binary_acumulator)
        return symbol_binary

    huffman_binary = [x[1] for x in HuffmanTree_to_HuffmanDictionary(HuffmanTree, '')]
    min_len = len(min(huffman_binary, key=len))
    max_len = len(max(huffman_binary, key=len))


    symbols = []
    def IsSymbol(tree, shorted_string):
        if isinstance(tree[int(shorted_string[0]) ^ 1], tuple):
            return IsSymbol(tree[int(shorted_string[0]) ^ 1], shorted_string[1:])
        else:
            symbols.append(tree[int(shorted_string[0]) ^ 1])
            return max_len - len(shorted_string) + 1

    # Huffman Decoder
    aux = 0
    while aux < len(band_binary):
        aux += IsSymbol(HuffmanTree, band_binary[aux:aux + max_len])
        max_len = min(max_len, len(band_binary) - aux)

    # Some cases this happens, so we can deal with that, later this must be gone!
    M, N = shapes
    if len(symbols) != int(M*N/2):
        varaux = int(M * N / 2) - len(symbols)
        for i in range(varaux):
            symbols.append(symbols[-1])

    # Degroup symbols
    band_degroup = np.zeros(shape=(M, N))
    for i in range(M):
        for j in range(int(N / 2)):
            band_degroup[i][j * 2] = int((symbols[i * int(N / 2) + j] / pow(2, 8)) // 1)
            band_degroup[i][j * 2 + 1] = symbols[i * int(N / 2) + j] - (band_degroup[i][j * 2] * pow(2, 8))

    return band_degroup - minval

def BandDequantifier(band, minband, scalefactor, rate):
    dq_minband = minband / pow(10, 5)
    dq_scalefactor = scalefactor / pow(10, 5)

    return ((band / (pow(2, rate) - 1)) * dq_scalefactor) - dq_minband

def Inverse_N_level_WaveletTransform(cA, original_shape, level, cH=None, cV=None, wf='bior3.9'):
    if cH is None and cV is None:
        cH = np.zeros_like(cA)
        cV = np.zeros_like(cA)
    cD = np.zeros_like(cA)

    len_HVN_leveli = [[original_shape[0], original_shape[1]]]
    for i in range(level):
        (len_HVN_leveli[i][0] + (len_HVN_leveli[i][0] % 2 > 0)) / 2 + 9
        len_HVN_leveli.append([int((len_HVN_leveli[i][0] + (len_HVN_leveli[i][0] % 2 > 0)) / 2 + 9), int((len_HVN_leveli[i][1] + (len_HVN_leveli[i][1] % 2 > 0)) / 2 + 9)])

    coefficients = [cA, (cH, cV, cD)]
    for i in range(level-1):
        HVD = np.zeros((len_HVN_leveli[-1-i-1][0], len_HVN_leveli[-1-i-1][1]))
        coefficients.append((HVD, HVD, HVD,))
    print(len_HVN_leveli)

    return pywt.waverec2(coefficients, wf)

def ReSample420(band):
    return np.kron(band, np.ones((2,2)))

def YCoCg_to_RGB(imY, imCo, imCg):
    # ----------------------------------------------------------------------------
    # YCoCg to RGB
    ## -128 to recover negative values :)
    # imCo -= 128
    # imCg -= 128

    imgR = np.round(1 * imY + 1 * imCo - 1 * imCg)
    imgG = np.round(1 * imY + 0 * imCo + 1 * imCg)
    imgB = np.round(1 * imY - 1 * imCo - 1 * imCg)

    # ----------------------------------------------------------------------------
    # 3D array RGB to IMAGE
    imageShape = (imY.shape[0], imY.shape[1], 3,)
    rebuiltRGB = np.zeros(imageShape, dtype=np.uint8)

    for i in range(imageShape[0]):
        for j in range(imageShape[1]):
            rebuiltRGB[i, j] = [imgR[i, j], imgG[i, j], imgB[i, j]]

    RGBreconstructed = Image.fromarray(rebuiltRGB, 'RGB')

    return RGBreconstructed

def image_decoder_main():
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkorange", "gray"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkolivegreen", "gray"])

    # Retrieve information from pickle files
    #picklefilenames = FindFiles()
    HuffmanTree_Q_A_imY, HuffmanCoded_Q_A_imY, HuffmanMin_Q_A_imY, Q_min_A_imY, Q_SF_A_imY, Q_A_imY_shape, imY_shape = recoverPickleData('A_imY.pkl')
    HuffmanTree_Q_H_imY, HuffmanCoded_Q_H_imY, HuffmanMin_Q_H_imY, Q_min_H_imY, Q_SF_H_imY, Q_H_imY_shape, _ = recoverPickleData('H_imY.pkl')
    HuffmanTree_Q_V_imY, HuffmanCoded_Q_V_imY, HuffmanMin_Q_V_imY, Q_min_V_imY, Q_SF_V_imY, Q_V_imY_shape, _ = recoverPickleData('V_imY.pkl')
    HuffmanTree_Q_A_imSCo, HuffmanCoded_Q_A_imSCo, HuffmanMin_Q_A_imSCo, Q_min_A_imSCo, Q_SF_A_imSCo, Q_A_imSCo_shape, imSCo_shape = recoverPickleData('A_imSCo.pkl')
    HuffmanTree_Q_A_imSCg, HuffmanCoded_Q_A_imSCg, HuffmanMin_Q_A_imSCg, Q_min_A_imSCg, Q_SF_A_imSCg, Q_A_imSCg_shape, imSCg_shape = recoverPickleData('A_imSCg.pkl')

    # Decode with: Improved Symbol Reduction Technique base Huffman coder for efficient entropy coding in the transform coders
    Q_A_imY = Inverse_ISRT_basedHuffman(HuffmanTree_Q_A_imY, HuffmanCoded_Q_A_imY, HuffmanMin_Q_A_imY, Q_A_imY_shape)
    Q_H_imY = Inverse_ISRT_basedHuffman(HuffmanTree_Q_H_imY, HuffmanCoded_Q_H_imY, HuffmanMin_Q_H_imY, Q_H_imY_shape)
    Q_V_imY = Inverse_ISRT_basedHuffman(HuffmanTree_Q_V_imY, HuffmanCoded_Q_V_imY, HuffmanMin_Q_V_imY, Q_V_imY_shape)

    Q_A_imSCo = Inverse_ISRT_basedHuffman(HuffmanTree_Q_A_imSCo, HuffmanCoded_Q_A_imSCo, HuffmanMin_Q_A_imSCo, Q_A_imSCo_shape)
    Q_A_imSCg = Inverse_ISRT_basedHuffman(HuffmanTree_Q_A_imSCg, HuffmanCoded_Q_A_imSCg, HuffmanMin_Q_A_imSCg, Q_A_imSCg_shape)

    # Decuantize bands
    A_imY = BandDequantifier(Q_A_imY, Q_min_A_imY, Q_SF_A_imY, 5)
    H_imY = BandDequantifier(Q_H_imY, Q_min_H_imY, Q_SF_H_imY, 5)
    V_imY = BandDequantifier(Q_V_imY, Q_min_V_imY, Q_SF_V_imY, 5)

    A_imSCo = BandDequantifier(Q_A_imSCo, Q_min_A_imSCo, Q_SF_A_imSCo, 4)
    A_imSCg = BandDequantifier(Q_A_imSCg, Q_min_A_imSCg, Q_SF_A_imSCg, 4)

    # Inverse N level Wavelet Transform in wavelet family 'bior3.9'
    wavelet_family = 'bior3.9'
    print(imY_shape)
    imY = Inverse_N_level_WaveletTransform(cA=A_imY, original_shape=imY_shape, level=3, cH=H_imY, cV=V_imY, wf=wavelet_family)
    imSCo = Inverse_N_level_WaveletTransform(cA=A_imSCo, original_shape=imSCo_shape, level=4, cH=None, cV=None, wf=wavelet_family)
    imSCg = Inverse_N_level_WaveletTransform(cA=A_imSCg, original_shape=imSCg_shape, level=4, cH=None, cV=None, wf=wavelet_family)

    imCo = ReSample420(imSCo)
    imCg = ReSample420(imSCg)

    reconstructedImage = YCoCg_to_RGB(imY, imCo, imCg)
    #reconstructedImage.show()
    reconstructedImage = reconstructedImage.save('./images/img_reconstructed.png', 'png')




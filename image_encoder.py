#
# ENCODER IMAGE FOR INICTEL PROJECT
#
# By: Leonardo Montoya Obeso
# Last update: april 2023

import sys
from PIL import Image
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors

def RGB_to_YCoCg(filename):

    # Try Open Image
    try:
        img = Image.open(filename, 'r')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Decompose in RGB color model
    np_img = np.array(img)
    print(sys.getsizeof(np_img))
    imR, imG, imB = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2]

    # ----------------------------------------------------------------------------
    # RGB to YCoCg
    imY  =   1 / 4 * imR + 1 / 2 * imG + 1 / 4 * imB
    imCo =   1 / 2 * imR +   0   * imG - 1 / 2 * imB
    imCg = - 1 / 4 * imR + 1 / 2 * imG - 1 / 4 * imB

    return imY, imCo, imCg

def SubSample420(imY, imCo, imCg):

    # Reshape pixels (v and h)
    imageShape = imY.shape
    if imageShape[0] % 2 == 1:
        imY = np.vstack((imY, imY[-1, ...]))
        imCo = np.vstack((imCo, imCo[-1, ...]))
        imCg = np.vstack((imCg, imCg[-1, ...]))

    if imageShape[1] % 4 != 0:
        for i in range(4 - imageShape[1] % 4):
            imY = np.hstack((imY, np.vstack(imY[..., -1])))
            imCo = np.hstack((imCo, np.vstack(imCo[..., -1])))
            imCg = np.hstack((imCg, np.vstack(imCg[..., -1])))

    # Subsample with 4:2:0 method
    imSCo = np.delete(np.delete(imCo, slice(0, -1, 2), axis=0), slice(0, -1, 2), axis=1)
    imSCg = np.delete(np.delete(imCg, slice(0, -1, 2), axis=0), slice(0, -1, 2), axis=1)

    return imSCo, imSCg

def N_level_WaveletTransform(imX, N, wavelet_model):

    # N level Wavelet Transform in model 'bior3.9'
    coeffs = pywt.wavedec2(imX, wavelet_model, level=N)
    return (coeffs[0], coeffs[1]) # (A, (H, V, D)))

def BandQuantifier(band, rate):
    # Quantize band, band.min, scale factor
    band_min_value = band.min()
    SF = band.max() # try later: SF = band.max + band.min

    return  (((band - band_min_value) / SF) * (pow(2, rate) - 1)).round(decimals=0), \
            (SF * pow(10, 5)).round(decimals=0), \
            (np.absolute(band_min_value) * pow(10, 5)).round(decimals=0)

def ISRT_basedHuffman(band):

    # Huffman Encoder
    def HuffmanTreeConstructor(Symbol_Probability_TuplesList):
        a = sorted(Symbol_Probability_TuplesList, key=lambda x:x[1])
        if len(a) != 2:
            return HuffmanTreeConstructor([*[((a[1][0], a[0][0],), a[1][1] + a[0][1],)], *a[2:]])
        else:
            return [((a[1][0], a[0][0],), a[1][1] + a[0][1])]

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

    def HuffmanEncoder(huffman_tree, symbols):
        huffman_dictionary = HuffmanTree_to_HuffmanDictionary(huffman_tree, '')
        huffman_symbols = [x[0] for x in huffman_dictionary]
        huffman_binary = [x[1] for x in huffman_dictionary]

        HuffmanCodedBand = ''
        for i in range(symbols.shape[0]):
            for j in range(symbols.shape[1]):
                HuffmanCodedBand += huffman_binary[huffman_symbols.index(symbols[i][j])]

        return HuffmanCodedBand

    # Reshape horizontal size if necessary
    if band.shape[1] % 2 == 1:
        band = np.hstack((band, np.vstack(band[..., -1])))

    # Take care of negative values
    HuffmanMinValue = np.absolute(band.min())
    band += HuffmanMinValue

    # Group Symbols
    bandGS = np.zeros((band.shape[0], int(band.shape[1] / 2)))  # Grouped Symbols Band
    for i in range(bandGS.shape[0]):
        for j in range(bandGS.shape[1]):
            bandGS[i, j] = band[i, 2 * j] * pow(2, 8) + band[i, 2 * j + 1]

    # Find unique symbols and each appereance count
    list_values, list_probabilities = np.unique(np.asarray(bandGS).ravel(), return_counts=True)
    list_tuple_values_probs = list(zip(list_values, list_probabilities / np.sum(list_probabilities)))

    # Huffman Tree
    HuffmanTree = HuffmanTreeConstructor(list_tuple_values_probs)[0][0]

    # Huffman Encoder
    HuffmanCodedBand = HuffmanEncoder(HuffmanTree, bandGS)

    return HuffmanTree, HuffmanCodedBand, HuffmanMinValue

def createPickleFile(filename, data):
    filename += '.pkl'
    output = open(filename, 'wb')

    # Pickle dictionary using protocol 0.
    # pickle.dump(huffman, output)

    # Pickle the list using the highest protocol available.
    for d in data:
        pickle.dump(d, output, -1)

    output.close()

def image_encoder_main(filename):
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkorange", "gray"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkolivegreen", "gray"])

    #recursionlimit = sys.getrecursionlimit()
    #sys.setrecursionlimit(2000)

    # RGB to YCoCg
    imY, imCo, imCg = RGB_to_YCoCg(filename)

    # Subsample chrominance bands with 4:2:0 model
    imSCo, imSCg = SubSample420(imY, imCo, imCg)

    # Apply N level Wavelet Transform in model 'bior3.9'
    wavelet_model = 'bior3.9'

    A_imY, (H_imY, V_imY, _) = N_level_WaveletTransform(imY, 3, wavelet_model)
    A_imSCo, _ = N_level_WaveletTransform(imSCo, 4, wavelet_model)
    A_imSCg, _ = N_level_WaveletTransform(imSCg, 4, wavelet_model)

    # Quantize Bands
    Q_A_imY, Q_SF_A_imY, Q_min_A_imY = BandQuantifier(A_imY, 5)
    Q_H_imY, Q_SF_H_imY, Q_min_H_imY = BandQuantifier(H_imY, 5)
    Q_V_imY, Q_SF_V_imY, Q_min_V_imY = BandQuantifier(V_imY, 5)

    Q_A_imSCo, Q_SF_A_imSCo, Q_min_A_imSCo = BandQuantifier(A_imSCo, 4)
    Q_A_imSCg, Q_SF_A_imSCg, Q_min_A_imSCg = BandQuantifier(A_imSCg, 4)

    # Encode with: Improved Symbol Reduction Technique base Huffman coder for efficient entropy coding in the transform coders
    HuffmanTree_Q_A_imY, HuffmanCoded_Q_A_imY, HuffmanMin_Q_A_imY = ISRT_basedHuffman(Q_A_imY)
    HuffmanTree_Q_H_imY, HuffmanCoded_Q_H_imY, HuffmanMin_Q_H_imY = ISRT_basedHuffman(Q_H_imY)
    HuffmanTree_Q_V_imY, HuffmanCoded_Q_V_imY, HuffmanMin_Q_V_imY = ISRT_basedHuffman(Q_V_imY)

    HuffmanTree_Q_A_imSCo, HuffmanCoded_Q_A_imSCo, HuffmanMin_Q_A_imSCo = ISRT_basedHuffman(Q_A_imSCo)
    HuffmanTree_Q_A_imSCg, HuffmanCoded_Q_A_imSCg, HuffmanMin_Q_A_imSCg = ISRT_basedHuffman(Q_A_imSCg)

    print(sys.getsizeof(HuffmanCoded_Q_A_imY) + sys.getsizeof(HuffmanCoded_Q_H_imY) + sys.getsizeof(HuffmanCoded_Q_V_imY) + sys.getsizeof(HuffmanCoded_Q_A_imSCo) + sys.getsizeof(HuffmanCoded_Q_A_imSCg))
    # Create Pickles Files
    createPickleFile('A_imY', (HuffmanTree_Q_A_imY, HuffmanCoded_Q_A_imY, HuffmanMin_Q_A_imY, Q_min_A_imY, Q_SF_A_imY, Q_A_imY.shape, imY.shape))
    createPickleFile('H_imY', (HuffmanTree_Q_H_imY, HuffmanCoded_Q_H_imY, HuffmanMin_Q_H_imY, Q_min_H_imY, Q_SF_H_imY, Q_H_imY.shape, imY.shape))
    createPickleFile('V_imY', (HuffmanTree_Q_V_imY, HuffmanCoded_Q_V_imY, HuffmanMin_Q_V_imY, Q_min_V_imY, Q_SF_V_imY, Q_V_imY.shape, imY.shape))

    createPickleFile('A_imSCo', (HuffmanTree_Q_A_imSCo, HuffmanCoded_Q_A_imSCo, HuffmanMin_Q_A_imSCo, Q_min_A_imSCo, Q_SF_A_imSCo, Q_A_imSCo.shape, imSCo.shape))
    createPickleFile('A_imSCg', (HuffmanTree_Q_A_imSCg, HuffmanCoded_Q_A_imSCg, HuffmanMin_Q_A_imSCg, Q_min_A_imSCg, Q_SF_A_imSCg, Q_A_imSCg.shape, imSCg.shape))

    #sys.setrecursionlimit(recursionlimit)




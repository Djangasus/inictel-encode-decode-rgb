# inictel-encode-decode-rgb

## Description

The `inictel-encode-decode-rgb` project focuses on the encoding and decoding of RGB images using the bior3.9 wavelet family and the ISRT Huffman Coder. These processes are fully detailed in [1].

Probably the one noteworthy component of this project is the developed ISRT Huffman Encoder [2]. It employs a straightforward recursive function to create the Huffman Tree and another function to encode data based on the generated tree. However, for larger images there may be considerations regarding recursion limits.

Also, this code successfully addresses the absence of wavelet-generated (A,(V,H,D)) matrices for the bior3.9 wavelet transformations, both for square, `anya.png`, and non-square, `lena.png`, images.

This code was developed during the first month of my internship at INICTEL-UNI.

## Bibliography

[1] K. Guerra, J. Casavilca, S. Huamán, et al., "A low-rate encoder for image transmission using LoRa communication modules," *International Journal of Information Technology*, vol. 15, pp. 1069–1079, 2023. [DOI: 10.1007/s41870-022-01077-7](https://doi.org/10.1007/s41870-022-01077-7)

[2] V. S. Thakur, K. Thakur, S. Gupta, "An improved symbol reduction technique based Huffman coder for efficient entropy coding in the transform coders," *IET Image Processing*, 2021; 15: 1008–1022. [DOI: 10.1049/ipr2.12081](https://doi.org/10.1049/ipr2.12081)

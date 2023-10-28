#
# MAIN FILE FOR ENCODER-DECODER RGB IMAGE - INICTEL PROJECT
#
# By: Leonardo Montoya Obeso
# Last update: april 2023

from image_encoder import *
from image_decoder import *

def main() -> None:
    filename = "./images/ania.png"

    image_encoder_main(filename)

    image_decoder_main()

if __name__ == '__main__':
    main()


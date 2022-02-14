"""
Script to perform the following data preparation steps:
- training data standardisation 
- feature extraction 
- data cleaning
- calculation of reference model 

"""

import sys
from time import time
import get_train_data
import retrieve_data
import extract_2d_features
import extract_3d_features
import clean_data
import get_ref_model


def main(params): 

    starttime = time()

    get_train_data.main(params)

    retrieve_data.main(params)

    extract_2d_features.main(params)

    extract_3d_features.main(params)
    
    clean_data.main(params)

    get_ref_model.main(params)

    endtime = time()
    duration = endtime - starttime
    print('\n>> Computation time: ', round(duration, 2), 's \n\n' + 10*'-')


if __name__ == '__main__':
    main(sys.argv[1])

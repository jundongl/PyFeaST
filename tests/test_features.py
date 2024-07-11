import pytest
from sklearn.feature_selection import mutual_info_classif
import numpy as np 

def test_feature_ranking():
    # Mock data for testing (Scrap data for test purpose)
    data = [
        [0.2, 0.5, 0.1, 0.8],
        [0.3, 0.7, 0.4, 0.6],
        [0.1, 0.9, 0.3, 0.2],
        [0.6, 0.4, 0.7, 0.3]
    ]
    
    # Mock labels/targets
    labels = [1, 0, 1, 0]  
    
    # Calculated  mutual information scores for each feature
    ranking = mutual_info_classif(data, labels)
    
    assert isinstance(ranking, np.ndarray)
    
    # these ranking returns a list of scores
    assert len(ranking) == len(data[0]) 
    
    #In Future add  more test cases as needed

if __name__ == "__main__":
    pytest.main()

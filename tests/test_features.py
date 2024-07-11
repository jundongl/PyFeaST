import pytest
from skfeature.function import feature_ranking

def test_feature_ranking():
    # Mock data for testing
    data = [
        [0.2, 0.5, 0.1, 0.8],
        [0.3, 0.7, 0.4, 0.6],
        [0.1, 0.9, 0.3, 0.2],
        [0.6, 0.4, 0.7, 0.3]
    ]
    
    labels = [1, 0, 1, 0]  # Mock labels/targets
    
    # Example test case
    ranking = feature_ranking.mutual_info_score(data, labels)
    assert isinstance(ranking, list)
    assert len(ranking) == len(data[0])  # Assuming ranking returns a list of scores
    
    # Add more test cases as needed

if __name__ == "__main__":
    pytest.main()

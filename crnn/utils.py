# utils.py
import editdistance

def cer(prediction: str, target: str) -> float:
    """
    Calculates the Character Error Rate (CER) between a prediction and a target string.

    Args:
        prediction (str): The predicted string.
        target (str): The ground truth string.

    Returns:
        float: The Character Error Rate. Returns 0.0 if target is empty and prediction is also empty,
               1.0 if target is empty but prediction is not.
    """
    if not target: # Handle empty target string
        return 0.0 if not prediction else 1.0 
    
    distance = editdistance.eval(prediction, target)
    return distance / len(target)

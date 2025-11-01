import cv2
import numpy as np

def obstacle_detection(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    
    # Calculate probabilistic histogram
    prob_hist = hist / np.sum(hist)
    
    # Calculate entropy of the image
    entropy = -np.sum(prob_hist * np.log2(prob_hist + 1e-8))
    
    # Selection of the threshold using entropy-based method
    max_entropy = 0
    threshold = 0
    for k in range(256):
        omega_k = np.sum(prob_hist[:k+1])
        omega_k_complement = 1 - omega_k
        
        # Check if omega_k_complement is close to zero, if so, assign a small value to prevent division by zero
        if np.isclose(omega_k_complement, 0):
            omega_k_complement = 1e-8
        
        # Add epsilon value to prevent logarithm of zero
        prob_hist_A = prob_hist[:k+1] / (omega_k + 1e-8)
        prob_hist_B = prob_hist[k+1:] / (omega_k_complement + 1e-8)
        
        # Calculate entropy of regions A and B
        H_A = -np.sum(prob_hist_A * np.log2(prob_hist_A + 1e-8))
        H_B = -np.sum(prob_hist_B * np.log2(prob_hist_B + 1e-8))
        
        # Calculate object function value
        O = H_A + H_B
        
        if O > max_entropy:
            max_entropy = O
            threshold = k
    
    # Apply thresholding to detect obstacles
    _, obstacle_image = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    
    return obstacle_image, threshold, entropy
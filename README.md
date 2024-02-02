# StressSense : Estimation of stress levels using PPG signals

StressSense is a health concerned research project for effective diagnosis of mental stress levels. It involves binary classification of mental stress, excluding the stress in rest condition.

### INTRODUCTION
- Stress has become a major concern in modern societies.
   
- It can lead to:
  
  a) Low Productivity

  b) Health Issues
  
  c) Increased Cardiac Disease risk
   
- This project aims to provide a novel approach in estimating stress using photo-plethysmo-graph (PPG) Signals.
   
- Why PPG?
  
  a) Non-Invasive Method

  b) Low Cost

  c) Easy to measure

### RELATED WORKS

![image](https://github.com/KD-Blitz/StressSense/assets/118080140/71a42a67-b839-412d-bb2c-ae9fce43477f)

### OBJECTIVES
- Implement EMD on PPG signal.
- Feature extraction from IMFs.
- Develop a ML model to estimate stress level 
- Evaluate the performance of classification model

### PROPOSED METHODOLOGY
Step 1: Collection of Required Data
- We have utilized publicly available MAUS dataset. (Link - [1])

![image](https://github.com/KD-Blitz/StressSense/assets/118080140/10caafc7-7a77-4184-9907-a69f42656a7b)

Step 2: Data Preprocessing
- Data segregation into low and high sections
- Data merging of all participants.
- Use of Standard scaling

Step 3: Emperical Mode Decomposition for Feature Extraction
- Decomposing ppg signals into Intrinsic Mode Functions(IMFs)
- Extracting statistical features namely, mean, minimum, maximum and skewness from the first two IMFs

![image](https://github.com/KD-Blitz/StressSense/assets/118080140/421f3d2d-fba2-46a6-a63b-e0a2556449a4)

Step 4: Training and Classification
- Kept test size as 30% of the labeled dataset
- Check Random Forest Classifier
- Implemented Support Vector Machine(SVM) for classification.
- Performed Regularization

### RESULTS & VISUALIZATIONS
Sample 'Low stress' PPG signal with ite IMFs -->
![image](https://github.com/KD-Blitz/StressSense/assets/118080140/fb0a48d7-63c1-4097-bd51-1c5141c5b540)

Sample 'High stress' PPG signal with ite IMFs -->
![image](https://github.com/KD-Blitz/StressSense/assets/118080140/41ee361e-9071-4e18-ab9a-3bdbc095f0d1)

Correlation Matrix -->
![image](https://github.com/KD-Blitz/StressSense/assets/118080140/dc3146e8-1820-409f-9844-6051712da2c0)

Performance Report -->
![image](https://github.com/KD-Blitz/StressSense/assets/118080140/95b6add9-75e9-4ee4-98b2-957cf6e1539e)

Confusion Matrix -->

![image](https://github.com/KD-Blitz/StressSense/assets/118080140/ff152121-bf9d-42b8-a54d-af44cf07a34c)

### CONCLUSIONS/OUTCOMES
- The assessment of mental stress is a critical area of research in the field of human-computer interaction.
- The EMD method was applied to decompose the PPG signals into Intrinsic Mode Functions (IMFs), and statistical features.
- A classification model based on Support Vector Machines (SVM) was trained on these features to classify the PPG signals into low stress and high stress categories.
- The results showed that the proposed approach achieved an accuracy of 82% for stress estimation using PPG signals. 
- The use of EMD and statistical features provided an effective way to capture the relevant information related to stress in the PPG signals.

### FUTURE SCOPE
- Try to explore new methods 
- Deploying the model

### REFERENCES
1) https://ieee-dataport.org/open-access/maus
2) D. Jaiswal, A. Chowdhury, D. Chatterjee, and R. Gavas, “Unobtrusive smart-watch based approach for assessing mental workload,” in Proc. IEEE Region 10th Symp., 2019, pp. 304–309
3) F. Schaule, J. O. Johanssen, B. Bruegge, and V. Loftness, “Employing consumer wearables to detect office workers’ cognitive load for interruption management,” Proc. ACM Interactive, Mobile, Wearable Ubiquitous Technol., vol. 2, no. 1, pp. 1–20, 2018
4) D. Ekiz, Y. S. Can, and C. Ersoy, “Long short-term network based unobtrusive perceived workload monitoring with consumer grade smartwatches in the wild,” IEEE Trans. Affect. Comput., p. 1, 2019, doi: 10.1109/TAFFC.2021.3110211
5) https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9664394&tag=1
6) https://www.mql5.com/en/forum/emd-imf
7) https://www.researchgate.net/Exploring_EEG_Empirical_Mode_Decomposition

### TECH STACKS INVOLVED
- python

### RESEARCH TECHNIQUES INVOLVED
- Symmetric Projection Attractor Reconstruction (SPAR)
- Cross Wavelet Transform (XWT)
- Chirp Z Transform (CZT)
- Heart Rate Variability (HRV) 
- Emperical Mode Decomposition (EMD)

# TEAM
Krishna Dubey (ML and Reasearch), Pankaj Kumar Giri (Reasearch)

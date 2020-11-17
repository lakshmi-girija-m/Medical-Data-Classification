# Medical Data Classification
### About Dataset: <h3>
Three medical datasets (csv files) were used in this project:
* diabetic_data.csv - Identifying patients with high risk of readmission within a period of 30 days
* rhc.csv - Identifying patients who have received RHC have survived or not
* vlbw.csv - Identifying infants with very low birth weight have survived or not

The fourth dataset is in prescription format. There were 100 to 120 prescriptions about medical history of patients. These prescriptions contain diseases, treatments and allergies related to the patients. A file named "Concepts" contains indices of medical terms in the prescription along with their tags (problem, treatment or test). A csv file was created with this information using nltk library. CRF and LSTM algorithm was used for classification of the data. More information about dataset can be obtained [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3168320/).

### Installing required librarires: <h3>
* Installing __keras__ and __tenserflow__:
```
conda install keras==2.2.4
conda install tensorflow==1.13.1
```
* Installing __keras-contrib__:
```
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python setup.py install
```
You can refer this [link](https://kegui.medium.com/how-to-install-keras-contrib-7b75334ab742) for detailed step by step process.
* Installing __seqeval__:
```
conda install -c conda-forge seqeval
```
__Note__: Run Anaconda Prompt as Administrator and execute these commands.

### Deatils of the Project: <h3>
This project evaluates performance of machine learning algorithms based on their F1 scores. Genetic algorithms technique was used for optimization. We observe that __Random Forest__ algorithm preforms well for all datsets. Measures for evaluating performance (for each dataset) were:
* F1 scores over 10 runs
 <div align="center">
  <img src="/Images/F1_1.png" height="300" width="390"><img src="/Images/F1_2.png" height="300" width="390"><img src="/Images/F1_3.png" height="300" width="390">
 </div>
 
 * F1 score by varying size of test set
<div align="center">
  <img src="/Images/Test_1.png" height="300" width="390"><img src="/Images/Test_2.png" height="300" width="390"><img src="/Images/Test_3.png" height="300" width="390">
</div>

* Effect of encoding on F1 score over 10 runs:
<div align="center">
  <img src="/Images/Encoding_1.png" height="300" width="390"><img src="/Images/Encoding_2.png" height="300" width="390"><img src="/Images/Encoding_3.png" height="300" width="390">
</div>
 
For prescrption dataset, CRF and LSTM algorithm was used for classification of the data. Medical terms in the prescriptions were classified into three classes: problem (P), treatment (M) and test (T).
<div align="center">
  <img src="/Images/NLP.png" height="200" width="590">
</div>

### Proposed Paper: <h3>
You can find the proposed paper [here](https://drive.google.com/file/d/1Qzbqyrg81OGvobxF7fjzUYCsVMD3UUdL/view?usp=sharing). This paper was presented at NCBICS 2019 held on 13 - 14th May 2019 at CMRIT.

# Medical Data Classification
##### Introduction: <h4>

#### About Dataset: <h4>
Three medical datasets (csv files) were used in this project:
* diabetic_data.csv - Identifying patients with high risk of readmission within a period of 30 days
* rhc.csv - Identifying patients who have received RHC have survived or not
* vlbw.csv - Identifying infants with very low birth weight have survived or not

The fourth dataset is in prescription format. There were 100 to 120 prescriptions about medical history of patients. These prescriptions contain diseases, treatments and allergies related to the patients. A csv file was created with this information using nltk library. CRF and LSTM algorithm was used for classification of the data. More information about dataset can be obtained in the following link:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3168320/

#### Installing required librarires: <h4>
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
  You can refer this [link](https://kegui.medium.com/how-to-install-keras-contrib-7b75334ab742)
  * Installing __seqeval__:
```
conda install -c conda-forge seqeval
```
__Note__: Run Anaconda Prompt as Administrator and execute these commands.

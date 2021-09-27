
0. Where to find the code/data?
https://github.com/SiyLiu/ML7641



1. About The Project
It is very crucial for financial institutes to predict the default status of credit card customers, which will help the bank take actions in time to keep the level of loss. Based upon experience, both basic information and the payment behaviors matter in predicting the default status.

This study sampled two datasets with the size of 3000 from the original datasets, and saved a separate data for the validation purposes (size: 2000). Two datasets keep partial features respectively. The first dataset keeps all personal information including limit balance, gender, education, marital status, and age, and payment status for the past 6 months. The second dataset includes the payment amount and bill statement amount in the past 6 months.


2. Getting Started
Prerequisites
Below lists the environment prerequisties for the project.

platform: win-64
ca-certificates=2021.7.5=haa95532_1
certifi=2021.5.30=py38haa95532_0
cycler=0.10.0=pypi_0
joblib=1.0.1=pypi_0
kiwisolver=1.3.2=pypi_0
matplotlib=3.4.3=pypi_0
numpy=1.21.2=pypi_0
openssl=1.1.1l=h2bbff1b_0
pandas=1.3.3=pypi_0
pillow=8.3.2=pypi_0
pip=21.0.1=py38haa95532_0
pyparsing=2.4.7=pypi_0
python=3.8.11=h6244533_1
python-dateutil=2.8.2=pypi_0
pytz=2021.1=pypi_0
scikit-learn=1.0=pypi_0
scipy=1.7.1=pypi_0
seaborn=0.11.2=pypi_0
setuptools=58.0.4=py38haa95532_0
six=1.16.0=pypi_0
sqlite=3.36.0=h2bbff1b_0
threadpoolctl=2.2.0=pypi_0
vc=14.2=h21ff451_1
vs2015_runtime=14.27.29016=h5e58377_2
wheel=0.37.0=pyhd3eb1b0_1
wincertstore=0.2=py38_0

Instruction
The data files(.csv) is saved under the same directory as the main program. To kick off the program, please open corresponding .ipynb file and hit run button above. The program will automatically train the model, conduct Grid Search, generate the plots, and dump the model into a file.
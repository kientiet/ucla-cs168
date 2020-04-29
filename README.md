# ucla-cs168

Here are the steps to run the code

1/ You can download the dataset here: [Kaggle](https://www.kaggle.com/joangibert/tcga_coad_msi_mss_jpg)

2/ you must place the zip file in the this folder. 

3/ You can run this code to extract the data on terminal: `sh scripts/install.sh`

* **Note**: Remember to use `cd` to navigate the terminal to this folder

4/ Then you can follow the instruction to change the name of the files accordingly.

5/ Eventually, you can run `sh scripts/split.sh` to split the file for convenience to read

6/ To run the experiments: You need to run `python3 install requirements.txt`
    Then you need to install Jupyter Notebook if you don't have.
* **Note**: All the code I wrote in python 3.7. So please, install python 3 if you are still using python 2
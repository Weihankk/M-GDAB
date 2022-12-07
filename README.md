> %% **Code and data are not currently publicly available as our studies has not yet been accepted for publishing.**

> %% **When our research is accepted for publication we make sure to release the code as soon as possible.**


<img src="https://github.com/Weihankk/M-GDAB/blob/main/logo.png" width="130px">

*A Multi network fusion framwork to predict gene-disease associations of brain*

---------------
# Machine required
- Due to the large scale of the network used in M-GDAB, with up to 550,000 edges, we have tried to run M-GDAB on Nvidia-Tesla-V100S-32G while be noticed that the memeory GPU is insufficient. So we imployed it on the fat nodes of CPU high performance computing(HPC) clusters. These nodes are equipped with Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz and 4TB physical memory. 

- **The monitoring system found that the maximum memory usage when running M-GDAB reached ~3.2Tb and lasts about 16 days before running complete! Therefore, please make sure that your HPC meets the resource requirements before running. It is not recommended to run M-GDAB on personal PCs.**

- For easy test, we provide a demo dataset with only a small amount of data, ensure anyone can quickly test and experience M-GDAB on any machine.

# Packages required 
- python - 3.7.11
- pytorch - 1.12.1
- nibabel - 4.0.2
- networkx - 2.6.3
- scipy - 1.7.3
- numpy - 1.21.6
- pandas - 1.3.5
- torch-geometric - 2.1.0
- tensorflow - 1.15.0
- scikit-learn - 1.0.2

# Get started
The M-GDAB is an end-to-end framework, which can input, parse, train and evaluate automated. It is easy to use.
## Clone M-GDAB on your machine

```
git clone https://github.com/Weihankk/M-GDAB.git

cd M-GDAB
```

> Raw data download shell scripts are all within ./Raw_data folder, Please ensure that you have permissions to the appropriate database and agree with the terms before downloading.

## Load and parse raw data
```
python Load_dataset.py
```

> This will create a folder named "Precess_data", which stored all the network files.

## Parallel speedup by splitting dataset per k-fold
```
python Split_dataset.py
```

> This will create a folder named "Split_data", which stored the 5-CV pre-split data.

## Running by each k-fold
```
python K1.py
python K2.py
python K3.py
python K4.py
python K5.py
```

> This will train and output the evaluation matrics of M-GDAB. You can deliver five tasks to different nodes at the same time to achieve multi-task acceleration

-------------
If you have any questions, please directly send an email to whzhang@webmail.hzau.edu.cn

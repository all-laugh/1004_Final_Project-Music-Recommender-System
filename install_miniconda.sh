wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${SCRATCH}/miniconda3

mkdir $SCRATCH/.cache
mkdir $SCRATCH/.conda

ln -s $SCRATCH/.conda ./home/${USER}/
ln -s $SCRATCH/.cache ./home/${USER}/

conda init bash

conda activate

conda create --name pyspark

pip install -r requirements.txt

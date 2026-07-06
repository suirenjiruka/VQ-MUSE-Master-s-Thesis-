
conda create -n VA_motion python==3.11
conda activate VA_motion
pip install -r requirements.txt

#add conda lib into env_paramenter while conda env activation
conda install -c conda-forge cudatoolkit=11.8 cudnn -y
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

#activation environmentt settup for tensorflow cudnn refernece
echo 'export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export CUDNN_PATH=$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#close cudnn synbol link while deacttivation
echo 'export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset CUDNN_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

#Run cuda check
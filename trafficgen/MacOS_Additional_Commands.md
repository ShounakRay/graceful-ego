# Info for the MacOS M1 chip

Once the conda environment has been installed, you'll get an error when running ```setup.py``` through ```pip install -e .```.
To get around this issue:

[Might not be necessary] Install GEOS manually through conda (since it won't work through PIP)
<https://gis.stackexchange.com/a/421566>
```bash
conda install -c conda-forge geos=3.7.1
```

Install shapely manually (since it won't work through PIP)
<https://shapely.readthedocs.io/en/latest/installation.html>
```bash
conda install --channel conda-forge shapely
```

[Only if necessary] Give proper permissions to your conda folder
<https://stackoverflow.com/a/68202088/9582712>
```bash
sudo chown -R $USER ~/.conda
```
echo $(pwd)

data_dir=data/
[ -d "$data_dir" ] || mkdir "$data_dir"

data_dir=$(pwd)/data
unzip tcga_coad_msi_mss_jpg.zip -d data_dir

echo "rename file data/MSIMUT_JPEG to msimut"
echo "rename file data/MSS_JPEG to mss"
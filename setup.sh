


rm -rf .python_dir
mkdir .python_dir
cd .python_dir
ln -s ../python classifier_tutorial


export PYTHONPATH=$PWD:$PYTHONPATH
cd ..

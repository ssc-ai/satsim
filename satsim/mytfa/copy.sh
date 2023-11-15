# in dir containing satsim/:
#   mkdir tensorflow
#   cd tensorflow
#   git clone https://github.com/tensorflow/addons.git

cp ../../../tensorflow/addons/tensorflow_addons/__init__.py .
mkdir -p image
cp -r ../../../tensorflow/addons/tensorflow_addons/image/__init__.py image/.
cp -r ../../../tensorflow/addons/tensorflow_addons/image/transform_ops.py image/.
cp -r ../../../tensorflow/addons/tensorflow_addons/image/translate_ops.py image/.
mkdir -p utils
cp -r ../../../tensorflow/addons/tensorflow_addons/utils/__init__.py utils/.
cp -r ../../../tensorflow/addons/tensorflow_addons/utils/types.py utils/.

# remove lines from __init__.py

# remove lines from image/__init__.py

# comment out 'tensorflow-addons<=0.19.0', in satsim/setup.py

TensorFlow Addons
=================

This code has been copied from the TensorFlow Addon repo which has been deprecated and is no longer maintained.

A few notes on what was copied:

Original Git Repo:

```
git clone https://github.com/tensorflow/addons.git
```

Copy files:

```
tensorflow_addons/__init__.py
tensorflow_addons/image/__init__.py
tensorflow_addons/image/transform_ops.py
tensorflow_addons/image/translate_ops.py
tensorflow_addons/image/utils.py
tensorflow_addons/utils/__init__.py
tensorflow_addons/utils/types.py
```

Edit files:

* remove lines from `__init__.py`
* remove lines from `image/__init__.py`
* replace `import tensorflow_addons` with `import satsim.tfa`

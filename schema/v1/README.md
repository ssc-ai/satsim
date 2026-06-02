SatSim JSON Schema
==================

This folder contains the v1 JSON schema. Root file is located in `Document.json`.
Reusable scalar and structured types are located in `types/`.

Requirements
------------

```
pip3 install json-schema-for-humans
npm install
```


Build HTML Representation
-------------------------

```
./build.sh
open build/index.html
```

Test Validate
-------------

```
npm run validate
```

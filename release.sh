#!/bin/bash

VER=0.1.0
DESC=FILL_IN_WITH_TEXT

cd dist
URL1=`curl -k --request POST --header "PRIVATE-TOKEN: $GITLAB_TOKEN" --form "file=@satsim-$VER-py2.py3-none-any.whl" https://gitlab.pacificds.com/api/v4/projects/347/uploads | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])"`
URL2=`curl -k --request POST --header "PRIVATE-TOKEN: $GITLAB_TOKEN" --form "file=@satsim-$VER.tar.gz" https://gitlab.pacificds.com/api/v4/projects/347/uploads | python3 -c "import sys, json; print(json.load(sys.stdin)['url'])"`
DATA=`echo \'{ "name": "SatSim $VER", "tag_name": "$VER", "description": "$DESC", "assets": { "links": [{ "name": "satsim-$VER-py2.py3-none-any.whl", "url": "https://gitlab.pacificds.com/machine-learning/satsim/$URL1" }, { "name": "satsim-$VER.tar.gz", "url": "https://gitlab.pacificds.com/machine-learning/satsim/$URL2" }] } }\'`
curl -k --header 'Content-Type: application/json' --header "PRIVATE-TOKEN: $GITLAB_TOKEN" --data $DATA --request POST https://gitlab.pacificds.com/api/v4/projects/347/releases

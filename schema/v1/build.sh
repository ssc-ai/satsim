#/bin/bash
rm -r build/
mkdir -p build/
generate-schema-doc --config-file config.yml Document.json build/index.html
generate-schema-doc --config-file config.yml --config template_name=md Document.json build/index.md
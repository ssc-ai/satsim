const fs = require('fs');
const util = require('util');
const Ajv = require('ajv');
const readFileAsync = util.promisify(fs.readFile);

function requireJSON(path) {
    return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function loadSchema(uri) {
    uri = uri.replace('https://ssc-ai.github.io/satsim/schema/v1/', './');
    console.log('Loading', uri);
    return readFileAsync(uri, 'utf8').then(JSON.parse);
}

var ajv = new Ajv({
    allErrors: true,
    verbose: true,
    loadSchema: loadSchema
});

var schema = requireJSON('./Document.json');

var filesToValidate = [
    '../../examples/sensors/jaws_sat_36411.0063_twobody_poppy.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063_ephemeris.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063_gc.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063_sgp4.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063_twobody_poppy_sprite.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063_twobody_poppy.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063_twobody.json',
    '../../examples/sensors/raven_rme02_sat_36411.0063.json',
    '../../examples/sensors/raven_rme02.json',
    '../../examples/sensors/raven_rme03.json',
    '../../examples/random/satsim.json',
    '../../examples/random/satsim_sstr7.json',
    '../../tests/config_dynamic_sstr7.json',
    '../../tests/config_dynamic.json',
    '../../tests/config_function.json',
    '../../tests/config_generator.json',
    '../../tests/config_import.json',
    '../../tests/config_none.json',
    '../../tests/config_piecewise.json',
    '../../tests/config_pipeline.json',
    '../../tests/config_poppy.json',
    '../../tests/config_static_sttr7_sgp4.json',
    '../../tests/config_static.json',
];

ajv.compileAsync(schema).then(validate => {

    filesToValidate.forEach(path => {
        console.log(`Validating ${path}...`);

        const packet = requireJSON(path);
        const valid = validate(packet);
        if (valid) {
            console.log(`...Valid!`);
        } else {
            console.log(`...Invalid: ${ajv.errorsText(validate.errors)}`);
        }
    });

}).catch(err => {
    console.log(err);
    console.log(err.stack);
});

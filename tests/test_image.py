"""Tests for `satsim.image.augment,model` package."""

import tensorflow as tf
import numpy as np

from satsim.image.augment import null, flip, crop_and_resize, scatter_shift_polar, pow, rotate, resize, scatter_shift_random, load_from_file
from satsim.image.model import astropy_model2d, sin2d, radial_cos2d, polygrid2d


def test_load_from_file():

    image = load_from_file(None, None, './tests/sprite.fits', True)
    expected = np.zeros((8,8))
    expected[3,3] = 0.33333334
    expected[4,3] = 0.33333334
    expected[4,4] = 0.33333334

    np.testing.assert_array_almost_equal(image, expected)


def test_resize():

    image = np.zeros((512,256))
    image[100,100] = 10.0

    out = resize(image, None, 256, 128)

    np.testing.assert_array_equal(out.numpy().shape, [256, 128])
    np.testing.assert_array_equal(tf.reduce_sum(out), tf.reduce_sum(image))

    out = resize(image, None, 128, 64, 4)

    np.testing.assert_array_equal(out.numpy().shape, [512, 256])
    np.testing.assert_array_equal(tf.reduce_sum(out), tf.reduce_sum(image))


def test_rotate():

    image = np.zeros((512,512))
    image[100,100] = 10.0

    expected = np.zeros((512,512))
    expected[100,100] = 10.0

    np.testing.assert_array_equal(rotate(image, 0.0, 0.0, 0.0), expected)

    expected = np.zeros((512,512))
    expected[411, 100] = 10.0

    np.testing.assert_array_equal(rotate(image, 0.0, 90.0, 0.0), expected)
    np.testing.assert_array_equal(rotate(image, 1.0, 0.0, 90.0), expected)


def test_pow():

    image = np.zeros((512,512))
    image[200,200] = 10.0

    expected = image ** 3
    np.testing.assert_array_equal(pow(image, None, 3.0, normalize=False, dtype=tf.float64), expected)

    np.testing.assert_array_equal(pow(image, None, 3.0, normalize=True, dtype=tf.float64), image)


def test_null():

    image = np.random.uniform(size=(512,512))
    np.testing.assert_array_equal(image, null(image, 0))


def test_flip():

    image = np.random.uniform(size=(512,512))
    np.testing.assert_array_equal(image, flip(image, 0, up_down=False, left_right=False))
    np.testing.assert_array_equal(np.flipud(np.fliplr(image)), flip(image, 0, up_down=True, left_right=True))


def test_crop_and_resize():

    image = tf.cast(np.random.uniform(size=(512,512)), tf.float32)

    image2 = crop_and_resize(image, 0, 0.0, 0.0, 1.0, 1.0)
    np.testing.assert_array_equal(image, image2)

    image3 = crop_and_resize(image, 0, 0.0, 0.0, 2.0, 2.0)
    np.testing.assert_array_equal(image, image3)

    image4 = np.random.uniform(size=(512,512))
    image4[0:257,0:257] = 1
    image4 = crop_and_resize(image4, 0, 0.0, 0.0, 0.5, 0.5)
    np.testing.assert_array_equal(np.ones((512,512)), image4)


def test_scatter_shift():

    image = np.zeros((512,512))
    image[200,200] = 1.0

    m = np.array([1.0])
    a = np.array([0.0])
    w = np.array([1.0])

    t_image = scatter_shift_polar(image, None, m, a, w, True, mode='shift')

    assert(t_image.numpy()[201,200] == 1)

    t_image = scatter_shift_polar(image, None, m, a, w, True, mode='shift', interpolation='bilinear')

    assert(t_image.numpy()[201,200] == 1)

    t_image = scatter_shift_polar(image, None, m, a, w, True, mode='roll')

    assert(t_image.numpy()[201,200] == 1)

    t_image = scatter_shift_polar(image, None, m, a, w, True, mode='fft')

    np.testing.assert_almost_equal(t_image.numpy()[201,200], 0.99999934, decimal=4)

    m = np.array([1.0])
    a = np.array([90.0])
    w = np.array([1.0])

    t_image = scatter_shift_polar(image, None, m, a, w, True, mode='shift')

    assert(t_image.numpy()[200,201] == 1)

    m = np.array([1.0, 2.0])
    a = np.array([90.0, 90.0])
    w = np.array([1.0, 1.0])

    t_image = scatter_shift_polar(image, None, m, a, w, True, mode='shift')

    assert(t_image.numpy()[200,201] == 0.5)
    assert(t_image.numpy()[200,202] == 0.5)

    image = np.zeros((512,512))
    image[200,200] = 1.0

    m = np.array([1.0])
    a = np.array([0.0])
    w = np.array([1.0])

    t_image = scatter_shift_random(image, None, 0.0, 1.0e-6, 5, mode='shift')

    assert(tf.reduce_sum(t_image) == tf.reduce_sum(tf.cast(image, tf.float32)))

    t_image = scatter_shift_random(image, None, 0.0, 1.0e-6, 0, mode='shift')

    np.testing.assert_array_equal(t_image.numpy(), image)


def test_sin2d():

    h = 512
    w = 256

    image = sin2d(h, w)
    assert(image.shape[0] == h)
    assert(image.shape[1] == w)
    assert(np.min(image) == 0)
    assert(np.max(image) == 1)

    image = sin2d(h, w, maximum=0.5)
    assert(np.min(image) == 0)
    assert(np.max(image) == 0.5)

    image = sin2d(h, w, bias=10)
    assert(np.min(image) == 9)
    assert(np.max(image) == 11)

    image = sin2d(h, w, direction=45, damped=True)
    assert(image.shape[0] == h)
    assert(image.shape[1] == w)
    assert(np.min(image) == 0)

    image = sin2d(h, w, frequency=0.0)
    assert(image.shape[0] == h)
    assert(image.shape[1] == w)
    np.testing.assert_array_equal(image, np.ones((h,w)))


def test_radial_cos2d():

    h = 513
    w = 257

    image = radial_cos2d(h, w, 1.0, 1.0, 1.0, clip=None)
    assert(image.shape[0] == h)
    assert(image.shape[1] == w)
    assert(image[0,128] == -1)
    assert(image[256,128] == 1)
    assert(image[256,256] == -1)

    image = radial_cos2d(h, w, 0.5, 0.5, 1.0)
    assert(image.shape[0] == h)
    assert(image.shape[1] == w)
    np.testing.assert_almost_equal(image[0,128], 0)
    assert(image[256,128] == 1)
    np.testing.assert_almost_equal(image[256,256], 0)

    image = radial_cos2d(h, w, power=1.0, xy_scale=0.5)
    assert(image.shape[0] == h)
    assert(image.shape[1] == w)
    np.testing.assert_almost_equal(image[0,128], 0)
    assert(image[256,128] == 1)
    np.testing.assert_almost_equal(image[256,256], 0)


def test_radial_polygrid2d():

    h = 512
    w = 256

    image = polygrid2d(h, w, [[17.560146642169087, 66.88358528391808, -1923.3437232582494, 26567.576463235335, -207001.21378903894, 971692.8001972447, -2779235.3612668547, 4590496.6343407035, -3371107.8756765523, -1048210.2589694241, 2698180.9282684117, 714642.2376605271, -1901133.9100504196, -1104580.694059071, 1059202.5743227832, 1320338.4705447122, -296183.92191669473, -1200879.4668746728, -44858.340995945364, 1076924.6385281654, -597373.5850815261, 94377.65713407454], [153.32899191518345, -761.7810714072812, 19036.180281640976, -198331.15107363628, 1194287.1572933893, -4560903.618771703, 10730640.92207711, -13401722.502179453, 3547797.5748655014, 9009399.675164133, -2943988.2699503265, -8033330.042930883, -969991.1394683055, 6239396.771219749, 4864492.62173233, -2036979.3896609028, -5774471.469800368, -2131646.1969824843, 4123245.943991525, 3807509.478865818, -4560001.450250363, 1076065.1094665318], [-5803.13998473715, -594.5910554421134, -72554.18821439822, 538902.1997813039, -1410905.0956253593, 881204.5092761964, 2553485.9390028547, -5402909.185980823, 3207710.382020113, 1464466.4678524577, -3090380.5794776175, -653407.5430916527, 3631265.78001488, 2585490.663978509, -2692582.5868631713, -5580890.997217878, -2051573.0817365083, 4795005.132415626, 7059453.276921695, -338524.91221234435, -10237634.651103135, 4825524.384674541], [117504.89386255387, 78375.14635325316, 68656.64290550048, -1447942.9468574007, 3799230.8920435524, -1601116.4508387244, -4134801.436778033, 2222540.163414357, 6323713.225680093, -3004527.8221042557, -7632746.419463046, -1540973.6463092559, 5671653.13728812, 6084083.676095725, 971441.4843968449, -3682797.758308633, -4012659.1824097764, -948348.8170233577, 1746170.3729067743, 1365710.1736207833, -863160.4725909424, 336463.60466890316], [-1269770.1910642735, -764850.6936053638, 761270.1677792147, 832669.5469818392, -5203953.526664703, 2535782.9541899096, -1496969.4720600438, 2320794.538634287, 6097824.278785949, 163328.76390350366, -3522330.5626527104, -1531321.703902828, 645691.5388351629, -43752.347387192305, -1682640.92789842, -1630833.7955567474, 188354.35707471543, 1860386.6075495845, 2081023.0998336386, 1614453.571308666, 1471316.5257320455, -2620477.0604424886], [8077223.238085874, 3951117.648947257, -2308664.124816529, 3720497.2299852665, 2020986.7674873269, 5433694.927422616, -4743086.892716728, -9000800.559315972, -4615045.968181518, -2322855.523567686, 175395.0611687075, 2301638.7213117485, 1735199.2473694645, -341239.8552115259, -1009184.532683749, 331052.8876870801, 1340140.7227972574, -290045.1276589617, -3664860.6992729474, -4696101.046540382, -811583.3116735676, 88519.85818893672], [-31366101.57701779, -12921608.576825704, -710410.3404688975, -6562410.460720489, -3383094.3435470313, 3887995.2883260776, 7004087.211331353, 4051187.7314259443, 5994790.3037836235, 7328173.902347655, 8026231.383217273, 6520335.416594372, 2546078.4889334743, -633279.9651490232, -40871.16961196915, 3354031.1902295123, 5600706.225664552, 3986638.3085425906, 357780.82640777656, 153897.11777160922, 6054707.1320582945, 7203279.575022821], [73072253.89528005, 26789782.557476193, 11456485.372312594, 2312980.389467407, -3378546.9802339077, -9387408.865459098, -3592929.7039382844, -8349042.975270483, -9966351.873445924, -9956057.541249786, -9045836.167014264, -9847543.278784141, -12597732.503408847, -13908400.35919092, -11236612.568804115, -6270086.948427622, -3372563.6585890856, -5083764.477966803, -9048103.735279594, -9645723.588385731, -5180021.782404673, -9411673.532586275], [-89777398.71619077, -29984663.15972119, -14057946.857103124, 4078681.959959676, 9013699.894729918, 1432886.301934808, 11811387.259798402, 9060590.584106926, 7661616.728803351, 8397736.496355169, 10697063.038400417, 11396826.566934383, 9753354.421103675, 8744900.003406601, 10716300.763921423, 14172427.267658522, 15331703.016885431, 12361635.809037974, 8022205.730628753, 7608290.495387601, 11471187.476694815, 3398468.8080274933], [21765949.555308808, 4443872.643028015, -3113200.855639931, -363796.8764044002, 2260106.801821265, -7885871.963848192, 1362875.8764387106, -1532171.5828261832, -3247564.0727778226, -2828320.648696173, -464210.3512439648, 648740.8386827015, -759393.3057400822, -2290953.560265491, -1720414.4741574933, -34034.475717520865, -132754.74061596836, -3073582.7892395216, -5860345.381474419, -3713944.220400298, 2516017.814400393, -4956810.226581441], [68182797.04252174, 24175408.90646, 10110229.79476191, -3135175.6299841525, -1897355.516191441, -10007262.753040863, -3286670.2356197955, -6514825.795076888, -8189025.392762477, -8411188.457385678, -6684408.3116512755, -5656605.556246844, -6914282.437250553, -8686660.509148503, -8961874.89208027, -8318330.139370485, -8936439.25065882, -11330072.086353043, -12479072.894401047, -8122698.657877931, 131693.8068733426, -5962642.26406922], [-29916069.53913847, -6481479.909864377, 4177034.2097155293, -3232491.6360884192, 2525136.93889751, 884595.2842223579, 5546115.038456418, 2950212.027836017, 2359071.6763490937, 1925723.7573352817, 3066957.453958878, 4114927.739389336, 3391588.0723244594, 2054093.8626711303, 1771226.848474688, 2131150.01919609, 1350610.939221411, -870423.3996045326, -1690807.9990253148, 2718691.1029205434, 10529346.540107328, 4143126.408682956], [-65249645.77539486, -22762538.835898943, -4302979.198340873, -2046225.0524254702, 4082740.7169803884, 6619090.986973501, 7853655.191580406, 5810333.059539084, 6572775.672929256, 6037727.430115296, 6566989.301559746, 7646840.891998188, 7655328.785771529, 7204062.609577766, 7566522.611379528, 8306232.319398675, 7743678.053028606, 5494759.701996822, 4054008.1567891287, 6920483.109771181, 12526734.061132615, 4856464.550195021], [11027055.9429124, -1316647.3834374156, -3372212.1995736235, 71679.65596247447, -681219.6354884554, 2003864.4106588527, -1199904.5314540893, -2848626.598351187, -860903.0609821336, -1682961.4885939565, -2021622.8644176032, -1236975.477405841, -804525.2469787569, -514324.82847240835, 599653.4092222601, 2033133.333276009, 2046739.4807932994, -27179.038876693347, -2157080.056263146, -1092953.2019241755, 2221281.063980862, -5964575.114419813], [66942604.62400955, 20265526.629727524, 1027196.5606937525, 2786419.331366881, -5067270.156672131, -2479279.1122639216, -8556926.547692692, -8995322.86095622, -5683551.636191978, -6859552.31779163, -8237502.430419571, -8152831.153301469, -7915663.620651948, -7530942.387338716, -6130200.012240185, -4175103.4925518525, -3481136.8640073594, -5086259.977219879, -7485038.114128481, -7624991.527313821, -5596769.803365399, -12449581.077156223], [20087527.085818704, 11342417.625395212, 737941.7725449863, 4601353.676878808, -4287387.511320089, 672995.0602560991, -5488903.090654984, -3744563.639732446, 921186.8891360534, -521691.58805324655, -2765351.663001948, -3449133.4527028007, -3852765.0346242767, -4057026.5939937616, -3038788.2714081006, -1013010.1913569728, 219729.58349469816, -748654.4126198781, -2963894.690432652, -3553054.618384881, -1645288.5058032232, -5371674.634661963], [-57869473.85579563, -12956499.978739198, -2560860.360702834, 3746191.7617924367, -1225455.966423655, 6787648.971334753, 1728854.4403374684, 5305618.499128299, 10622304.269719278, 8926380.197351143, 6383453.919592177, 5451380.212765905, 4527239.9699998405, 3467969.205909118, 3663616.6773134777, 5369603.770870153, 6938148.670887149, 6634335.509245426, 4848267.311594513, 4250952.321794521, 6676209.803644131, 6948526.208902616], [-49070145.31181945, -16978143.649861407, -1742505.268934504, 117400.3178989042, -1131067.543091234, 6860332.611754864, 1651985.1589545417, 5177346.079460803, 9667875.286642618, 7585216.515888075, 5600296.433311671, 5466678.388792626, 4729474.484829528, 3070899.8047621287, 2387519.146761611, 3600134.9120918363, 5349063.375183144, 5617325.6583503485, 4201959.150125512, 3499205.991556269, 6210844.460556237, 10006564.202532396], [44003385.49504993, 6159765.44929697, 3914695.0373362335, -3774016.2215432636, -3401089.5609011897, 501091.67349217273, -6289999.881482561, -4637768.669030302, -2389151.8367372416, -4951769.918846979, -5450713.064087168, -3604900.6760132513, -3259076.4952212875, -5056578.397311994, -6500132.326371667, -5847222.270703974, -4064633.7982603214, -3436769.371210682, -4829685.575105603, -6258143.435735633, -4340844.356558578, 1199966.6222828762], [66804070.952357434, 18106673.03919011, 4664810.815906825, -4175681.5013117185, -685914.5156513087, -1450254.2752723163, -8038832.540999725, -7847108.304310266, -8187142.664855864, -11169094.58701544, -9613354.284300858, -5133382.675641255, -3287799.2325174925, -5103452.30624617, -7374179.631977869, -7406201.8869230915, -5713924.765673936, -4890078.769515177, -6600296.818306096, -9487363.407692779, -9777530.591814278, -4940282.494728199], [-77194663.88738872, -13987552.288261266, -6017962.95649693, 76649.17729943723, 9092082.028774023, 6659899.597263708, 6077205.594874221, 7972427.197143596, 5704509.585779163, 1965372.19821823, 4974576.187169146, 11381391.991375418, 13975906.798377873, 11485113.08671512, 7990129.931295757, 7241838.883174093, 9171787.145300552, 10751346.98830116, 9281071.838054374, 5167051.159617513, 2174502.847567641, 4610090.246525197], [21638654.735443067, 2842390.1748507256, 1323696.97846248, 2018417.6893986156, -1883516.6793615075, -10051889.388223086, -404550.59698934347, 5745293.69013849, 867762.3063814815, -6155743.529020745, -4815953.258876292, 533629.6532651649, 1499521.0220554771, -3101436.2113559567, -8107960.145961672, -8666077.300272958, -4516817.34609799, 612816.6141437129, 2612929.9780205972, 472702.55532959686, -2592090.21405257, -1111115.1025875136]])

    assert(image.shape[0] == h)
    assert(image.shape[1] == w)


def test_astropy_model2d():

    h = 512
    w = 256

    image = astropy_model2d(h, w, './tests/stray_light.asdf')

    assert(image.shape[0] == h)
    assert(image.shape[1] == w)

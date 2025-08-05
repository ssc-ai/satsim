from datetime import timezone

import numpy as np

from astropy.time import Time
from astropy import units as u

from skyfield.constants import AU_KM
from skyfield.units import _ltude
from skyfield.vectorlib import VectorFunction

from satsim import time
from satsim.geometry.astrometric import load_earth, eci_to_ecr
from satsim.geometry.sgp4 import create_sgp4
from satsim.vecmath import Quaternion

from czmlpy.core import Document, Packet, Preamble
from czmlpy.enums import InterpolationAlgorithms, ReferenceFrames, ExtrapolationTypes
from czmlpy.properties import (
    Billboard,
    Clock,
    Color,
    Label,
    Material,
    Path,
    Position,
    SolidColorMaterial,
    RectangularSensor,
    Orientation,
    Double,
)
from czmlpy.types import IntervalValue, TimeInterval
from erfa import gd2gc


def save_czml(ssp, obs_cache, astrometrics, filename):
    """Saves scenario as a Cesium czml file.

    Args:
        ssp: `dict`, SatSim input configuration.
        obs_cache: `list`, list of SatSim Skyfield objects.
        filename: `str`, if not None, save czml output. default=None

    Returns:
        A `dict`, the CZML output
    """

    if 'czml_samples' in ssp['sim']:
        N = ssp['sim']['czml_samples']
    else:
        N = 10

    if 'time' in ssp['geometry']:
        tt = ssp['geometry']['time']
    else:
        tt = [2020, 1, 1, 0, 0, 0.0]

    exposure_time = ssp['fpa']['time']['exposure']
    frame_time = ssp['fpa']['time']['gap'] + exposure_time
    num_frames = ssp['fpa']['num_frames']

    t0 = time.utc_from_list(tt)
    t2 = time.utc_from_list(tt, frame_time * num_frames)

    extractor = CZMLExtractor(time.to_astropy(t0), time.to_astropy(t2))

    # extract site data
    if 'site' in ssp['geometry']:
        site = ssp['geometry']['site']
        name = site['name'] if 'name' in site else None

        # czml
        czml_config = site.get('czml', {})
        label_show = czml_config.get('label_show', False)
        cone_show = czml_config.get('cone_show', True)
        cone_color = czml_config.get('cone_color', [255, 255, 0, 64])
        billboard_show = czml_config.get('billboard_show', True)
        billboard_image = czml_config.get('billboard_image', PIC_GROUNDSTATION)

        sensor = {
            'y_fov': astrometrics[0]['y_fov'],
            'x_fov': astrometrics[0]['x_fov'],
            'time': [x['time'] for x in astrometrics],
            'quat': [list(_equatorial_to_ecr_quaternion(x['ra'], x['dec'], 0.0, x['time'])) for x in astrometrics],
            'range': [x['range'] for x in astrometrics],
        }

        if 'tle' in site:
            sat = create_sgp4(site['tle'][0], site['tle'][1])
            extractor.add_space_station(sat, sensor, label_text=name, label_show=label_show,
                                        cone_show=cone_show, cone_color=cone_color,
                                        billboard_show=billboard_show, billboard_image=billboard_image)
        elif 'tle1' in site:
            sat = create_sgp4(site['tle1'], site['tle2'])
            extractor.add_space_station(sat, sensor, label_text=name, label_show=label_show,
                                        cone_show=cone_show, cone_color=cone_color,
                                        billboard_show=billboard_show, billboard_image=billboard_image)
        else:
            latitude = _ltude(site['lat'], 'latitude', 'N', 'S') * u.deg
            longitude = _ltude(site['lon'], 'longitude', 'E', 'W') * u.deg
            alt = site['alt'] * u.km
            extractor.add_ground_station([latitude, longitude, alt], sensor, label_text=name, label_show=label_show,
                                         cone_show=cone_show, cone_color=cone_color, billboard_show=billboard_show,
                                         billboard_image=billboard_image)

    # extract objects data
    for i, o in enumerate(ssp['geometry']['obs']['list']):

        ts_start_ob = t0
        ts_end_ob = t2
        if 'events' in o:
            if 'create' in o['events']:
                ts_start_ob = time.utc_from_list_or_scalar(o['events']['create'], default_t=tt)
            if 'delete' in o['events']:
                ts_end_ob = time.utc_from_list_or_scalar(o['events']['delete'], default_t=tt)

        if obs_cache[i] is not None:
            sat = obs_cache[i][0]

            name = o['name'] if 'name' in o else None

            start_interval = time.to_astropy(t0)
            end_interval = time.to_astropy(t2)

            # czml
            czml_config = o['czml'] if 'czml' in o else {}
            label_show = czml_config.get('label_show', False)
            path_show = czml_config.get('path_show', (ts_start_ob == t0))
            path_color = czml_config.get('path_color', [255, 255, 0])
            billboard_show = czml_config.get('billboard_show', True)
            billboard_image = czml_config.get('billboard_image', PIC_OBSERVATION if o['mode'] == 'observation' else PIC_SATELLITE)

            start_interval = time.to_astropy(time.utc_from_list_or_scalar(o['czml']['start_interval'])) if 'start_interval' in czml_config else time.to_astropy(t0)
            end_interval = time.to_astropy(time.utc_from_list_or_scalar(o['czml']['end_interval'])) if 'end_interval' in czml_config else time.to_astropy(t2)

            extractor.add_object(sat, N=N, id_name=name, label_text=name, start_interval=start_interval, end_interval=end_interval,
                                 start_available=time.to_astropy(ts_start_ob), end_available=time.to_astropy(ts_end_ob),
                                 label_show=label_show, path_show=path_show, path_color=path_color, billboard_show=billboard_show,
                                 billboard_image=billboard_image)

    # convert document to json
    j = extractor.get_document().dumps()
    if filename is not None:
        with open(filename, "w") as outfile:
            outfile.write(j)

    return j


PIC_SATELLITE = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAX"
    "NSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAADJSURBVD"
    "hPnZHRDcMgEEMZjVEYpaNklIzSEfLfD4qNnXAJSFWfhO7w2Zc0Tf9QG2rXrEzSUeZLOGm47W"
    "oH95x3Hl3jEgilvDgsOQUTqsNl68ezEwn1vae6lceSEEYvvWNT/Rxc4CXQNGadho1NXoJ+9i"
    "aqc2xi2xbt23PJCDIB6TQjOC6Bho/sDy3fBQT8PrVhibU7yBFcEPaRxOoeTwbwByCOYf9VGp"
    "1BYI1BA+EeHhmfzKbBoJEQwn1yzUZtyspIQUha85MpkNIXB7GizqDEECsAAAAASUVORK5CYI"
    "I="
)
PIC_GROUNDSTATION = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAX"
    "NSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAACvSURBVD"
    "hPrZDRDcMgDAU9GqN0lIzijw6SUbJJygUeNQgSqepJTyHG91LVVpwDdfxM3T9TSl1EXZvDwi"
    "i471fivK73cBFFQNTT/d2KoGpfGOpSIkhUpgUMxq9DFEsWv4IXhlyCnhBFnZcFEEuYqbiUlN"
    "wWgMTdrZ3JbQFoEVG53rd8ztG9aPJMnBUQf/VFraBJeWnLS0RfjbKyLJA8FkT5seDYS1Qwyv"
    "8t0B/5C2ZmH2/eTGNNBgMmAAAAAElFTkSuQmCC"
)
PIC_OBSERVATION = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACX"
    "BIWXMAAA7DAAAOwwHHb6hkAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48Gg"
    "AAAqFJREFUSIm1lbtPFFEUxn/nBnbvrmijBVCIUTEx0ZBAgz3GSixYNsbY8g/4YhMbQixopM"
    "ZHCcTMzBZCBwUtiSIFJJsYjY9CiLHzNeyGeyxmcFkdZnUTvvLc73zfOfdxLhwypCmjbHtwOo"
    "zIhfZ2HQSo1WQV1U2MLDASfmjNwMsPgJtCGIojW5ms6wCo7phvQFccX8aYEiM/XiXJmL8iih"
    "DYB4h7gXARlXsopxkNu3t6a2s9vbU1RsNuDGdAxoE+nHuJbyfRZjsygcG38/hW8e0MHh37l3"
    "srstJbkZXGTunAt4/xrRLYuXSTwD7At4qXu5u0fK7C2LkKY4m5fm48LmwyWdzLD+Bbh29n0p"
    "pMRdTJLuV8/15o3xm4KeAzGt5p2SAT3ga+4NxUo0HZ9iAMofKQIt9aNrjGV5BpYAgvd7Ju4H"
    "Q4YmiQKhDYMQKbfAZ72HUBIOCG6wYiF4BPFMN3qcnKDZQbqZzrO2+B7ViTNoD2dr0koh2nKr"
    "IcK3mvz/OkoXJHEegDwLPLGDwK4W9OdLukCPD+jcujMlilfshNn8j/QN2faoF9hG8/Nc307Q"
    "q+XfkH3tbedY86UN0Euijb06mJwjzCfCrHy54FOhHdqBsYWQAUJ4XU5EL4ZP++J8MUIi2zGN"
    "W0B98uAX1kwrPRfW4Bsxwja98A64yGV+odABhTAk5QtQ9bEgewdho4jphSMsG3k9HAyo0nLa"
    "cOOy9XiofdxMEVRH/BXEx8zHOO7l9OHNezHCOwTw8a120NZEHR8CaBfQvcp2qv4cs0uy6IX2"
    "gdz7JnMGYU0VsoxxEmGQknELRR8iCU8/3xVLwcR7YzWXdEgdqO+Q50xvElxJQo/FhPkmn+fr"
    "3cSYy7isrFTDb69Ks7soroBs4sUvz5sanGYeIXrs/2tlFecHUAAAAASUVORK5CYII="
)


def _equatorial_to_ecr_quaternion(ra, dec, roll=0.0, time=None):
    """ Converts ECI RA/Dec to ECR Quaternion """
    ra, dec, roll = eci_to_ecr(time, ra, dec, roll)

    hpr = {
        'pitch': dec.to(u.rad).value,
        'heading': ra.to(u.rad).value,
        'roll': roll
    }

    q = Quaternion.fromHeadingPitchRoll(hpr)

    return [q.x, q.y, q.z, q.w]


class CZMLExtractor:
    """A class for extracting SatSim data to Cesium"""

    def __init__(self, start_epoch, end_epoch, multiplier=60):
        """
        Orbital constructor

        Args:
            start_epoch: `astropy.time.core.Time`, Starting epoch
            end_epoch: `astropy.time.core.Time`, Ending epoch
        """
        self.packets = []
        self.num_objects = 0
        self.num_gs = 0

        self.start_epoch = Time(start_epoch, format="isot")
        self.end_epoch = Time(end_epoch, format="isot")

        pckt = Preamble(
            name="document_packet",
            clock=IntervalValue(
                start=self.start_epoch,
                end=self.end_epoch,
                value=Clock(
                    currentTime=self.start_epoch.datetime.replace(tzinfo=timezone.utc),
                    multiplier=multiplier,
                ),
            ),
        )
        self.packets.append(pckt)

    def add_ground_station(
        self,
        pos,
        sensor,
        id_name=None,
        id_description=None,
        label_fill_color=[255, 255, 0, 255],
        label_outline_color=[255, 255, 0, 255],
        label_font="16pt Lucida Console",
        label_text=None,
        label_show=False,
        billboard_show=True,
        billboard_image=PIC_GROUNDSTATION,
        cone_color=[255, 255, 0, 64],
        cone_show=True,
    ):
        """
        Adds a ground station

        Args:
            pos: `list [~astropy.units]`, coordinates of ground station (i.e. latitude, longitude, altitude)
            id_description: `str`, Set ground station description
            label_fill_color: `list (int)`, Fill Color in rgba format
            label_outline_color: `list (int)`, Outline Color in rgba format
            label_font: `str`, Set label font style and size (CSS syntax)
            label_text: `str`, Set label text
            label_show: `bool`, Indicates whether the label is visible
        """
        lat, lon, alt = pos
        pos = list(gd2gc(1, lon.to_value(u.rad), lat.to_value(u.rad), alt.to_value(u.m)))

        gs_id = "GS" + str(self.num_gs)
        pckt = Packet(
            id=gs_id,
            name=id_name,
            description=id_description,
            availability=TimeInterval(start=self.start_epoch, end=self.end_epoch),
            position=Position(cartesian=pos),
            label=Label(
                show=True,
            ),
            billboard=Billboard(image=billboard_image, show=True),
        )

        self.packets.append(pckt)
        self.num_gs += 1

        N = len(sensor['quat'])
        quat = []
        radius = []
        for i, t, q, r in zip(range(N), sensor['time'], sensor['quat'], sensor['range']):
            s = (Time(t) - self.start_epoch).to(u.second).value
            quat += [s, q[0], q[1], q[2], q[3]]
            radius += [s, r * 1000]

        # test add sensor
        pckt = Packet(
            id=gs_id + "_FOV",
            parent=gs_id,
            position=Position(reference=gs_id + "#position"),
            orientation=Orientation(
                unitQuaternion=quat,
                interpolationDegree=1,
                interpolationAlgorithm=InterpolationAlgorithms.LINEAR,
                epoch=self.start_epoch.datetime.replace(tzinfo=timezone.utc),
                forwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE,
                backwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE
            ),
            agi_rectangularSensor=RectangularSensor(
                show=cone_show,
                showIntersection=False,
                intersectionColor=Color(rgba=[255, 255, 255, 255]),
                intersectionWidth=2,
                portionToDisplay="COMPLETE",
                lateralSurfaceMaterial=Material(solidColor=SolidColorMaterial(color=Color(rgba=cone_color))),
                domeSurfaceMaterial=Material(solidColor=SolidColorMaterial(color=Color(rgba=cone_color))),
                xHalfAngle=np.radians(sensor['x_fov'] / 2),
                yHalfAngle=np.radians(sensor['y_fov'] / 2),
                radius=Double(
                    number=radius,
                    interpolationDegree=1,
                    interpolationAlgorithm=InterpolationAlgorithms.LINEAR,
                    epoch=self.start_epoch.datetime.replace(tzinfo=timezone.utc),
                    forwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE,
                    backwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE
                )
            )
        )
        self.packets.append(pckt)

    def add_space_station(
        self,
        sat,
        sensor,
        id_name=None,
        id_description=None,
        label_fill_color=[255, 255, 0, 255],
        label_outline_color=[255, 255, 0, 255],
        label_font="16pt Lucida Console",
        label_text=None,
        label_show=False,
        billboard_show=True,
        billboard_image=PIC_GROUNDSTATION,
        cone_color=[255, 255, 0, 64],
        cone_show=True,
        path_color=[255, 255, 0],
        path_show=True,
    ):
        """Adds a space-based observing platform"""

        if not isinstance(sat, VectorFunction):
            return

        sat = sat - load_earth()

        gs_id = "GS" + str(self.num_gs)

        cart = []
        quat = []
        radius = []
        for t, q, r in zip(sensor['time'], sensor['quat'], sensor['range']):
            ts = Time(t)
            sec = (ts - self.start_epoch).to(u.second).value
            position, _, _, _ = sat._at(time.from_astropy(ts))
            position = position * AU_KM * 1000
            cart += [sec, position[0], position[1], position[2]]
            quat += [sec, q[0], q[1], q[2], q[3]]
            radius += [sec, r * 1000]

        pckt = Packet(
            id=gs_id,
            name=id_name,
            description=id_description,
            availability=TimeInterval(start=self.start_epoch, end=self.end_epoch),
            position=Position(
                interpolationDegree=1,
                interpolationAlgorithm=InterpolationAlgorithms.LINEAR,
                referenceFrame=ReferenceFrames.INERTIAL,
                cartesian=cart,
                epoch=self.start_epoch.datetime.replace(tzinfo=timezone.utc),
            ),
            path=Path(
                show=path_show,
                width=2,
                material=Material(solidColor=SolidColorMaterial(color=Color.from_list(path_color))),
                resolution=120,
            ),
            label=Label(show=True),
            billboard=Billboard(image=billboard_image, show=billboard_show),
        )
        self.packets.append(pckt)
        self.num_gs += 1

        pckt = Packet(
            id=gs_id + "_FOV",
            parent=gs_id,
            position=Position(reference=gs_id + "#position"),
            orientation=Orientation(
                unitQuaternion=quat,
                interpolationDegree=1,
                interpolationAlgorithm=InterpolationAlgorithms.LINEAR,
                epoch=self.start_epoch.datetime.replace(tzinfo=timezone.utc),
                forwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE,
                backwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE,
            ),
            agi_rectangularSensor=RectangularSensor(
                show=cone_show,
                showIntersection=False,
                intersectionColor=Color(rgba=[255, 255, 255, 255]),
                intersectionWidth=2,
                portionToDisplay="COMPLETE",
                lateralSurfaceMaterial=Material(solidColor=SolidColorMaterial(color=Color(rgba=cone_color))),
                domeSurfaceMaterial=Material(solidColor=SolidColorMaterial(color=Color(rgba=cone_color))),
                xHalfAngle=np.radians(sensor['x_fov'] / 2),
                yHalfAngle=np.radians(sensor['y_fov'] / 2),
                radius=Double(
                    number=radius,
                    interpolationDegree=1,
                    interpolationAlgorithm=InterpolationAlgorithms.LINEAR,
                    epoch=self.start_epoch.datetime.replace(tzinfo=timezone.utc),
                    forwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE,
                    backwardExtrapolationType=ExtrapolationTypes.EXTRAPOLATE,
                ),
            ),
        )
        self.packets.append(pckt)

    def add_object(
        self,
        sat,
        N=10,
        id_name=None,
        id_description=None,
        path_width=None,
        path_show=None,
        path_color=[255, 255, 0],
        label_fill_color=[255, 255, 0, 255],
        label_outline_color=[255, 255, 0, 255],
        label_font="16pt Lucida Console",
        label_text=None,
        label_show=False,
        start_interval=None,
        end_interval=None,
        start_available=None,
        end_available=None,
        billboard_show=True,
        billboard_image=PIC_SATELLITE,
    ):
        """
        Adds a SatSim Skyfield object

        Args:
            sat: `object`, SatSim Skyfield object
            N: `int`, Number of sample points
            id_name: `str`, Set orbit name
            id_description: `str`, Set orbit description
            path_width: `int`, Path width
            path_show: `bool`, Indicates whether the path is visible
            path_color: `list (int)`, Rgba path color
            label_fill_color: `list (int)`, Fill Color in rgba format
            label_outline_color: `list (int)`, Outline Color in rgba format
            label_font: `str`, Set label font style and size (CSS syntax)
            label_text: `str`, Set label text
            label_show: `bool`, Indicates whether the label is visible
        """

        # ignore non-skyfield objects
        if not isinstance(sat, VectorFunction):
            return

        sat = sat - load_earth()  # adjust from barycenter to earth centered

        cart_cords = []

        tt = time.linspace(time.from_astropy(start_interval), time.from_astropy(end_interval), N)
        ss = np.linspace(0, (end_interval - start_interval).to(u.second).value, N)

        for ts, sec in zip(tt, ss):
            position, _, _, _ = sat._at(ts)
            position = position * AU_KM * 1000
            cart_cords += [sec, position[0], position[1], position[2]]

        cartesian_cords = cart_cords

        start_epoch = Time(start_interval, format="isot")

        path = None
        if path_show:
            path = Path(
                show=None,
                width=path_width,
                material=Material(solidColor=SolidColorMaterial(color=Color.from_list(path_color))),
                resolution=120,
            )

        pckt = Packet(
            id=self.num_objects,
            name=id_name,
            description=id_description,
            availability=TimeInterval(start=start_available, end=end_available),
            position=Position(
                interpolationDegree=3,
                interpolationAlgorithm=InterpolationAlgorithms.LAGRANGE,
                referenceFrame=ReferenceFrames.INERTIAL,
                cartesian=cartesian_cords,
                # Use explicit UTC timezone, rather than the default, which is a local timezone.
                epoch=start_epoch.datetime.replace(tzinfo=timezone.utc),
            ),
            path=path,
            label=Label(
                text=label_text,
                font=label_font,
                show=label_show,
                fillColor=Color(rgba=label_fill_color),
                outlineColor=Color(rgba=label_outline_color),
            ),
            billboard=Billboard(image=billboard_image, show=billboard_show),
        )

        self.packets.append(pckt)

        self.num_objects += 1

    def get_document(self):
        """
        Retrieves CZML document.

        Returns:
            A `Document`, the CZML document.
        """
        return Document(self.packets)

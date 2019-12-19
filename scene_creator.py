import json
import typing as t
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import requests


def tuple_to_str(a_tuple: t.Tuple[t.Any]) -> str:
    return str(tuple(a_tuple)).replace("(", "").replace(")", "")


def tuple_to_color(a_tuple: t.Tuple[float]) -> str:
    r, g, b = np.clip(np.asarray(a_tuple), 0, 255).astype(np.uint8)
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b).lower()


def tuple_coordinates_to_dict(a_tuple: t.Tuple[float]) -> t.Dict[str, str]:
    return {
        "x": str(a_tuple[0]),
        "y": str(a_tuple[1]),
        "z": str(a_tuple[2]),
    }


class SceneCreator:
    def __init__(self):
        self.scene = ET.Element("scene", attrib={"version": "0.6.0"})

        light = ET.SubElement(
            self.scene, "emitter", attrib={"type": "constant"}
        )
        radiance = ET.SubElement(
            light, "spectrum", attrib={"name": "radiance", "value": "1"}
        )
        # self.add_ground()
        self.add_integrator()

        self.elements = [light, radiance]

    def add_perspective_camera(
        self,
        lookat: t.Tuple[int],
        transform_name: str,
        origin: t.Tuple[int],
        up: t.Tuple[int],
    ):
        camera = ET.SubElement(
            self.scene, "sensor", attrib={"type": "perspective"}
        )
        transform = ET.SubElement(
            camera, "transform", attrib={"name": transform_name}
        )
        ET.SubElement(
            transform,
            "lookat",
            attrib={
                "origin": tuple_to_str(origin),
                "target": tuple_to_str(lookat),
                "up": tuple_to_str(up),
            },
        )

        sampler = ET.SubElement(camera, "sampler", {"type": "ldsampler"})
        ET.SubElement(
            sampler, "integer", {"name": "sampleCount", "value": "32"}
        )

    def add_sphere(
        self, center: t.Tuple[int], radius: float, color: t.Tuple[float]
    ):
        sphere = ET.SubElement(self.scene, "shape", attrib={"type": "sphere"})
        point = ET.SubElement(
            sphere,
            "point",
            attrib={
                "name": "center",
                "x": str(center[0]),
                "y": str(center[1]),
                "z": str(center[2]),
            },
        )
        radius = ET.SubElement(
            sphere, "float", attrib={"name": "radius", "value": str(radius)}
        )
        bsdf = ET.SubElement(sphere, "bsdf", attrib={"type": "diffuse"})
        srgb = ET.SubElement(
            bsdf,
            "srgb",
            attrib={"name": "reflectance", "value": tuple_to_color(color)},
        )
        self.elements.extend([sphere, radius, point, bsdf, color])

    def add_ground(self):
        disk = ET.SubElement(self.scene, "shape", attrib={"type": "disk"})
        transform = ET.SubElement(
            disk, "transform", attrib={"name": "toWorld"}
        )
        scale = ET.SubElement(transform, "scale", attrib={"value": "1000"})
        translate = ET.SubElement(transform, "translate", attrib={"z": "-0.2"})
        bsdf = ET.SubElement(disk, "bsdf", attrib={"type": "diffuse"})
        srgb = ET.SubElement(
            bsdf,
            "srgb",
            attrib={
                "name": "reflectance",
                "value": tuple_to_color((255, 255, 255)),
            },
        )

    def add_integrator(self):
        integrator = ET.SubElement(
            self.scene, "integrator", {"type": "volpath"}
        )

    def add_cylinder(
        self,
        p0: t.Tuple[float],
        p1: t.Tuple[float],
        radius: float,
        color: t.Tuple[int],
    ):
        cylinder = ET.SubElement(
            self.scene, "shape", attrib={"type": "cylinder"}
        )

        p0 = ET.SubElement(
            cylinder,
            "point",
            attrib={"name": "p0", **tuple_coordinates_to_dict(p0)},
        )

        p1 = ET.SubElement(
            cylinder,
            "point",
            attrib={"name": "p1", **tuple_coordinates_to_dict(p1)},
        )
        bsdf = ET.SubElement(cylinder, "bsdf", attrib={"type": "diffuse"})
        srgb = ET.SubElement(
            bsdf,
            "srgb",
            attrib={"name": "reflectance", "value": tuple_to_color(color)},
        )
        radius = ET.SubElement(
            cylinder, "float", attrib={"name": "radius", "value": str(radius)}
        )

        self.elements.extend([cylinder, p0, p1, bsdf, srgb, radius])

    def __str__(self):
        return ET.tostring(self.scene)

    def __repr__(self):
        return ET.tostring(self.scene)

    def to_str(self) -> str:
        return ET.tostring(self.scene)


def render(xml_data: str) -> np.ndarray:
    result = requests.post("http://localhost:8000/render", data=xml_data)
    data = json.loads(result.content)
    data = np.asarray(data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

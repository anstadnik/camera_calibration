from dataclasses import dataclass

from PIL import Image


@dataclass
class Corner:
    camera_id: int
    corner_id: int
    marker_id: int
    x: float
    y: float


@dataclass
class Entry:
    name: str
    width: int
    height: int
    num_corners: int
    encoding: str
    corners: list[Corner]
    image: Image.Image | None = None

    def __init__(self, orpc_path: str):
        with open(orpc_path) as f:
            lines = f.readlines()
            # TODO: Use iterator
            self.name = lines[0].split(":")[1].strip()
            self.width = int(lines[1].split(":")[1].strip())
            self.height = int(lines[2].split(":")[1].strip())
            self.num_corners = int(lines[3].split(":")[1].strip())
            self.encoding = lines[4].split(":")[1].strip()
            self.corners = []
            for row in lines[5:]:
                data = row.strip().split(",")
                corner = Corner(
                    int(data[0]),
                    int(data[1]),
                    int(data[2]),
                    float(data[3]),
                    float(data[4]),
                )
                self.corners.append(corner)



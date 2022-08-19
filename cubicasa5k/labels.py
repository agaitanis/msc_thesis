import numpy as np
from enum import IntEnum

class Label(IntEnum):
    BACKGROUND = 0
    ROOM = 1
    WALL = 2
    DOOR = 3

_SVG_LABEL_TO_LABEL = {
    "Alcove": Label.ROOM,
    "Attic": Label.ROOM,
    "Ballroom": Label.ROOM,
    "Bar": Label.ROOM,
    "Basement": Label.ROOM,
    "Bath": Label.ROOM,
    "Bedroom": Label.ROOM,
    "Below150cm": Label.ROOM,
    "CarPort": Label.ROOM,
    "Church": Label.ROOM,
    "Closet": Label.ROOM,
    "ConferenceRoom": Label.ROOM,
    "Conservatory": Label.ROOM,
    "Counter": Label.ROOM,
    "Den": Label.ROOM,
    "Dining": Label.ROOM,
    "DraughtLobby": Label.ROOM,
    "DressingRoom": Label.ROOM,
    "EatingArea": Label.ROOM,
    "Elevated": Label.ROOM,
    "Elevator": Label.ROOM,
    "Entry": Label.ROOM,
    "ExerciseRoom": Label.ROOM,
    "Garage": Label.ROOM,
    "Garbage": Label.ROOM,
    "Hall": Label.ROOM,
    "HallWay": Label.ROOM,
    "HotTub": Label.ROOM,
    "Kitchen": Label.ROOM,
    "Library": Label.ROOM,
    "LivingRoom": Label.ROOM,
    "Loft": Label.ROOM,
    "Lounge": Label.ROOM,
    "MediaRoom": Label.ROOM,
    "MeetingRoom": Label.ROOM,
    "Museum": Label.ROOM,
    "Nook": Label.ROOM,
    "Office": Label.ROOM,
    "OpenToBelow": Label.ROOM,
    "Outdoor": Label.ROOM,
    "Pantry": Label.ROOM,
    "Reception": Label.ROOM,
    "RecreationRoom": Label.ROOM,
    "RetailSpace": Label.ROOM,
    "Room": Label.ROOM,
    "Sanctuary": Label.ROOM,
    "Sauna": Label.ROOM,
    "ServiceRoom": Label.ROOM,
    "ServingArea": Label.ROOM,
    "Skylights": Label.ROOM,
    "Stable": Label.ROOM,
    "Stage": Label.ROOM,
    "StairWell": Label.ROOM,
    "Storage": Label.ROOM,
    "SunRoom": Label.ROOM,
    "SwimmingPool": Label.ROOM,
    "TechnicalRoom": Label.ROOM,
    "Theatre": Label.ROOM,
    "Undefined": Label.ROOM,
    "UserDefined": Label.ROOM,
    "Utility": Label.ROOM,

    "Wall": Label.WALL,
    "Railing": Label.WALL,
    "Window" : Label.WALL,

    "Door" : Label.DOOR,
}


def get_label(s):
    return _SVG_LABEL_TO_LABEL[s]


def get_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)

    colormap[Label.BACKGROUND] = [255, 255, 255]
    colormap[Label.ROOM] = [255, 204, 153]
    colormap[Label.WALL] = [0, 0, 0]
    colormap[Label.DOOR] = [0, 255, 0]

    return colormap
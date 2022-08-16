import numpy as np

_SVG_LABEL_TO_COMMON_LABEL = {
    "Alcove": "Room",
    "Attic": "Room",
    "Ballroom": "Room",
    "Bar": "Room",
    "Basement": "Room",
    "Bath": "Room",
    "Bedroom": "Room",
    "Below150cm": "Room",
    "CarPort": "Room",
    "Church": "Room",
    "Closet": "Room",
    "ConferenceRoom": "Room",
    "Conservatory": "Room",
    "Counter": "Room",
    "Den": "Room",
    "Dining": "Room",
    "DraughtLobby": "Room",
    "DressingRoom": "Room",
    "EatingArea": "Room",
    "Elevated": "Room",
    "Elevator": "Room",
    "Entry": "Room",
    "ExerciseRoom": "Room",
    "Garage": "Room",
    "Garbage": "Room",
    "Hall": "Room",
    "HallWay": "Room",
    "HotTub": "Room",
    "Kitchen": "Room",
    "Library": "Room",
    "LivingRoom": "Room",
    "Loft": "Room",
    "Lounge": "Room",
    "MediaRoom": "Room",
    "MeetingRoom": "Room",
    "Museum": "Room",
    "Nook": "Room",
    "Office": "Room",
    "OpenToBelow": "Room",
    "Outdoor": "Room",
    "Pantry": "Room",
    "Reception": "Room",
    "RecreationRoom": "Room",
    "RetailSpace": "Room",
    "Room": "Room",
    "Sanctuary": "Room",
    "Sauna": "Room",
    "ServiceRoom": "Room",
    "ServingArea": "Room",
    "Skylights": "Room",
    "Stable": "Room",
    "Stage": "Room",
    "StairWell": "Room",
    "Storage": "Room",
    "SunRoom": "Room",
    "SwimmingPool": "Room",
    "TechnicalRoom": "Room",
    "Theatre": "Room",
    "Undefined": "Room",
    "UserDefined": "Room",
    "Utility": "Room",

    "Wall": "Wall",
    "Railing": "Wall",
    "Window" : "Wall",

    "Door" : "Door",
}

_COMMON_LABEL_TO_LABEL = {
    "Background" : 0,
    "Room" : 1,
    "Wall" : 2,
    "Door" : 3,
}

_COMMON_LABEL_TO_COLOR = {
    "Background" : [255, 255, 255],
    "Room" : [255, 204, 153],
    "Wall" : [50, 50, 50], 
    "Door" : [0, 255, 0],
}


def get_label(s):
    if s in _COMMON_LABEL_TO_LABEL:
        return _COMMON_LABEL_TO_LABEL[s]
    else:
        return _COMMON_LABEL_TO_LABEL[_SVG_LABEL_TO_COMMON_LABEL[s]]


def get_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)

    for common_label, label in _COMMON_LABEL_TO_LABEL.items():
        colormap[label] = _COMMON_LABEL_TO_COLOR[common_label]

    return colormap
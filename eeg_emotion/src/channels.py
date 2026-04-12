"""EEG channel metadata from the competition dataset description."""

CHANNEL_NAMES = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FT7",
    "FC3",
    "FCZ",
    "FC4",
    "FT8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "OZ",
    "O2",
)

CHANNEL_TO_INDEX = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

LEFT_RIGHT_CHANNEL_PAIRS = (
    ("FP1", "FP2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FT7", "FT8"),
    ("FC3", "FC4"),
    ("T3", "T4"),
    ("C3", "C4"),
    ("TP7", "TP8"),
    ("CP3", "CP4"),
    ("T5", "T6"),
    ("P3", "P4"),
    ("O1", "O2"),
)

LEFT_RIGHT_CHANNEL_INDICES = tuple(
    (CHANNEL_TO_INDEX[left], CHANNEL_TO_INDEX[right])
    for left, right in LEFT_RIGHT_CHANNEL_PAIRS
)

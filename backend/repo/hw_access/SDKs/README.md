# Vendor Hardware SDKs (Not Included)

This directory does not contain vendor SDK files.
The hardware SDKs used by RamanCopilot are proprietary software
provided by their respective manufacturers and are not redistributed
in this repository.

The virtual environment (`RAMAN_VIRTUAL_HW=1`, `python run_virtual.py`) does not need any of these files.

## Required SDKs

| Device | SDK | Required file(s) | Source |
|---|---|---|---|
| Andor iDus DV420 CCD | Andor SDK2 (2.104.30065.0) | `atmcd64d.dll`, `pyAndorSDK2` | Oxford Instruments (Andor) |
| TUCSEN Michrome 6 sCMOS | TUCam SDK | `TUCam.py`, `TUCam.dll` | TUCSEN |
| Märzhäuser Tango stage | Tango DLL (tested with 1.5.2.0) | `Tango_DLL.dll` | Märzhäuser Wetzlar  |

## Directory layout

Place the files so that this directory looks like this:

```
SDKs/
├── Tango_DLL.dll              
├── TuCam/
│   ├── __init__.py
│   ├── TUCam.py
│   └── lib/
│       └── x64/
│           └── TUCam.dll
└── andor_codes/
    ├── __init__.py
    ├── atmcd64d.dll
    └── pyAndorSDK2/
```
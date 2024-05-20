import exifread
import numpy as np 
from PIL import Image

file_name = "/home/p51pro/UD/jayraman_lab/Drive_dump/Sintered NCT Data/Apr 2023 Plate 1/4degC-0.82MPa-10min-charged/10,000rcf - 4degC - 10min (1CJT020)/50,000x mag/1CJT020 10000g CS1_060.tif"

im = Image.open(file_name)
na = np.array(im)

Image.fromarray(na).show()

f = open(file_name, 'rb')

# Return Exif tags
tags = exifread.process_file(f)

# Print the tag/ value pairs
    meterprint("Key: %s, value %s" % (tag, tags[tag]))
"""
TIFFReadDirectory: Warning, Unknown field with tag 34682 (0x877a) encountered.
TIFFFetchNormalTag: Warning, ASCII value for tag "Tag 34682" does not end in null byte. Forcing it to be null.
=== TIFF directory 0 ===
TIFF Directory at offset 0x8 (8)
  Subfile Type: (0 = 0x0)
  Image Width: 1024 Image Length: 943
  Resolution: 67, 67 pixels/cm
  Bits/Sample: 8
  Compression Scheme: None
  Photometric Interpretation: RGB color
  Samples/Pixel: 3
  Rows/Strip: 1
  Planar Configuration: single image plane
  Tag 34682: [User]
Date=03/14/2022
Time=11:52:46 AM
User=FIBuser
UserText=Helios
UserTextUnicode=480065006C0069006F007300

[System]
Type=DualBeam
Dnumber=D0521
Software=3.8.9.1943
BuildNr=1943
Source=FEG
Column=Elstar
FinalLens=Elstar
Chamber=xT-SDB
Stage=6inch
Pump=TMP
ESEM=no
Aperture=AVA
Scan=PIA 1.0
Acq=PIA 1.0
EucWD=0.004
SystemType=Helios NanoLab
DisplayWidth=0.320
DisplayHeight=0.240

[Beam]
HV=5000
Spot=
StigmatorX=-0.0130203
StigmatorY=0.33132
BeamShiftX=0
BeamShiftY=0
ScanRotation=0
ImageMode=Normal
Beam=EBeam
Scan=EScan

[EBeam]
Source=FEG
ColumnType=Elstar
FinalLens=Elstar
Acq=PIA 1.0
Aperture=AVA
ApertureDiameter=7.26825e-009
HV=5000
HFW=2.56e-006
VFW=2.21e-006
WD=0.00409547
BeamCurrent=8.59375e-011
TiltCorrectionIsOn=yes
DynamicFocusIsOn=yes
ScanRotation=0
LensMode=Immersion
SemOpticalMode=
ImageMode=Normal
SourceTiltX=0.015625
SourceTiltY=0.0449219
StageX=0.0113813
StageY=-0.0104484
StageZ=0.00398338
StageR=-2.10548
StageTa=0.90756
StageTb=0
StigmatorX=-0.0130203
StigmatorY=0.33132
BeamShiftX=0
BeamShiftY=0
EucWD=0.004
EmissionCurrent=
TiltCorrectionAngle=0
WehneltBias=

[GIS]
Number=4

[GIS1]
GasType=
HeaterState=
NeedleState=
GasFlow=
Port=G1

[GIS2]
GasType=IEE
HeaterState=Off
NeedleState=Retracted
GasFlow=Off
Port=G2

[GIS3]
GasType=
HeaterState=
NeedleState=
GasFlow=
Port=

[GIS4]
GasType=
HeaterState=
NeedleState=
GasFlow=
Port=

[Scan]
InternalScan=true
Dwelltime=3e-006
PixelWidth=2.5e-009
PixelHeight=2.5e-009
HorFieldsize=2.56e-006
VerFieldsize=2.21e-006
Average=0
Integrate=1
FrameTime=11.2712

[EScan]
Scan=PIA 1.0
InternalScan=true
Dwell=3e-006
PixelWidth=2.5e-009
PixelHeight=2.5e-009
HorFieldsize=2.56e-006
VerFieldsize=2.21e-006
FrameTime=11.2712
LineTime=0.012624
Mainslock=On
LineIntegration=4
ScanInterlacing=1

[Stage]
StageX=0.0113813
StageY=-0.0104484
StageZ=0.00398338
StageR=-2.10548
StageT=0.90756
StageTb=0
SpecTilt=0
WorkingDistance=0.00409547

[Image]
DigitalContrast=1
DigitalBrightness=0
DigitalGamma=1
Average=0
Integrate=1
ResolutionX=1024
ResolutionY=884
ZoomFactor=1
MagCanvasRealWidth=0.12800
MagnificationMode=3

[Vacuum]
ChPressure=0.000126093
Gas=
UserMode=High vacuum
Humidity=

[Specimen]
Temperature=

[Detectors]
Number=1
Name=TLD
Mode=SE

[TLD]
Contrast=66.0285
Brightness=38.9892
Signal=SE
ContrastDB=44.2215
BrightnessDB=-2.64274
SuctionTube=70
Mirror=-15
MinimumDwellTime=1e-007

[Accessories]
Number=0

[PrivateFei]
BitShift=0
DataBarSelected=HV curr tilt mag HFW WD Label MicronBar
DataBarAvailable=HV srot frame dwell curr WD mag HFW x y tilt filter det DateTime mode zoom Label MicronBar
TimeOfCreation=14.03.2022 11:52:46
DatabarHeight=59




"""

#!/bin/bash
# ----------------------------------------------------------------------------------
# User-defined parameters
# ----------------------------------------------------------------------------------
# Data directory
datadir="${HOME}/build/ufo-bundle/oops/qg/test/Data"

# File base
filebase="truth.fc"

# Date list
declare -a datelist=("2009-12-15T00:00:00Z.PT0S"
                     "2009-12-15T00:00:00Z.P1D"
                     "2009-12-15T00:00:00Z.P1DT12H"
                     "2009-12-15T00:00:00Z.P2D"
                     "2009-12-15T00:00:00Z.P2DT12H"
                     "2009-12-15T00:00:00Z.P3D"
                     "2009-12-15T00:00:00Z.P3DT12H"
                     "2009-12-15T00:00:00Z.P4D"
                     "2009-12-15T00:00:00Z.P4DT12H"
                     "2009-12-15T00:00:00Z.P5D"
                     "2009-12-15T00:00:00Z.P5DT12H"
                     "2009-12-15T00:00:00Z.P6D"
                     "2009-12-15T00:00:00Z.P6DT12H"
                     "2009-12-15T00:00:00Z.P7D"
                     "2009-12-15T00:00:00Z.P7DT12H"
                     "2009-12-15T00:00:00Z.P8D"
                     "2009-12-15T00:00:00Z.P8DT12H"
                     "2009-12-15T00:00:00Z.P9D"
                     "2009-12-15T00:00:00Z.P9DT12H"
                     "2009-12-15T00:00:00Z.P10D"
                     "2009-12-15T00:00:00Z.P10DT12H"
                     "2009-12-15T00:00:00Z.P11D"
                     "2009-12-15T00:00:00Z.P11DT12H"
                     "2009-12-15T00:00:00Z.P12D"
                     "2009-12-15T00:00:00Z.P12DT12H"
                     "2009-12-15T00:00:00Z.P13D"
                     "2009-12-15T00:00:00Z.P13DT12H"
                     "2009-12-15T00:00:00Z.P14D"
                     "2009-12-15T00:00:00Z.P14DT12H"
                     "2009-12-15T00:00:00Z.P15D"
                     "2009-12-15T00:00:00Z.P15DT12H"
                     "2009-12-15T00:00:00Z.P16D"
                     "2009-12-15T00:00:00Z.P16DT12H"
                     "2009-12-15T00:00:00Z.P17D"
                     "2009-12-15T00:00:00Z.P17DT12H"
                     "2009-12-15T00:00:00Z.P18D"
                    )

# Gif option
generate_gif=true
# ----------------------------------------------------------------------------------
# End of user-defined parameters
# ----------------------------------------------------------------------------------
# Export parameters
export datadir=${datadir}
export filebase=${filebase}
export difference=false
export generate_gif=${generate_gif}

# Run plotQGFields.sh
datelist="${datelist[@]@Q}" ./plotQGFields.sh
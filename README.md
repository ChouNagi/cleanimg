# cleanimg
A horrifically inefficient script that restores transparency to images.

Essentially starts off with some pixels that it thinks should be fully transparent, and then explores adjacent pixels until it finds a sufficiently opaque border and stops.

### Required Python Modules

PIL - `pip install Pillow`  
apng - `pip install apng`  

### Usage

`cleanimg [OPTIONS] inputfile outputfile`

### Options

#### -c / --colour / --color `COLOUR`
Specifies the colour of the background you want to remove.
All pixels in the picture of that colour will become transparent in the first iteration.
If not specified, and no guide colour is specified, the pixels in the image's edges of the most common colour will be inferred as the starting set.

#### -g / --guide `COLOUR`
Specifies a colour to act as a guide.
Pixels of the guide colour become the starting set.
This can (and should ideally) be different to the background colour.
Allows you to mark areas that should be transparent with magenta dots for instance.

#### -b / --border `COLOUR`
Specifies the expected colour of the border, allowing for more precise calculation of partially opaque pixels.
If omitted, the script will attempt to infer the border colour.

#### -a / --artefact / --artifact / --min `FLOAT`
Specifies the minimum opacity threshold threshold, with 0 = fully transparent, and 1 = fully opaque.
Encountering a pixel with transparency less than or equal to the threshold will cause it to be treated as fully transparent.
Useful for removing jpeg artifacts.
Defaults to 0.05

#### -t / --threshold / --max `FLOAT`
Specifies the maximum opacity threshold threshold, with 0 = fully transparent, and 1 = fully opaque.
Encountering a pixel with calculated transparency greater than or equal to the threshold will cause it to be treated as fully opaque and prevent exploration to neighbouring pixels.
Defaults to 0.8

#### -o / --overwrite
If not set, attempting to output to a file that already exists will prompt the user if they wish to overwrite it
If set, the prompt will be skipped, and the output file silently overwritten

#### -r / --resume `STRING`
Supplies an id for this run, and saves intermediate frame images to the work directory as it goes, allowing the conversion to be resumed.
Supplying the same resume id as a previous abortedly resumable invocation will cause it to resume.

#### -p / --optimise / --optimize
optimizes animations by merging identical frames and combining their durations, resulting in a smaller image

#### -v / --verbose
Generates additional text output about its current progress in the conversion

#### -d / --debug
Debugs the iteration process, generating an image for every step in the process with the current pixels highlighted in red in `cleanimg_DEBUG`
Useful for locating problematic regions in an image that won't convert properly.

### Colour Options

Note that options expecting a colour can take most:

* 3 or 6 character hexadecimal with or without preceding hash symbol
* rgb(R, G, B) colour
* a small subset of named colours like "black" or "white"

## Examples

`cleanimg -c white -b black inputfile outputfile`  
Treats all white pixels as transparent, and expects the borders to be black

`cleanimg -b black -t 0.9 inputfile outputfile`  
Scans inwards from the edge looking for a black border, with more tolerance than usual


## Note on animated Input Files

Background colour will be inferred on a per-frame basis if unspecified.  
For images with multiple frames, only outputting to PNG or GIF is supported.

## Note on Outputting to GIF format

GIF images do not support partial transparency. A pixel is either transparent or fully opaque.
If you wish for there to be partial transparency, make sure the output file is a PNG

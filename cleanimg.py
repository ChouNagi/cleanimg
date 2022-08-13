
# TL Note: It's pronounced GIF, not GIF

import os
import re
import sys
import math
import uuid
import argparse
try:
	from PIL import Image
except ImportError:
	print('Python Image Library (Pillow) is required to run this script')
	print('You can install it with: "pip install Pillow"')
	sys.exit(1)
try:
	from apng import APNG
except ImportError:
	print('APNG is required to run this script')
	print('You can install it with: "pip install apng"')
	sys.exit(1)

CURRENT_FOLDER = os.getcwd()
SCRIPTID = str(uuid.uuid4()).replace('-', '')
SCRIPT_PATH = sys.argv[0]
SCRIPT_FOLDER = os.path.abspath(os.path.join(SCRIPT_PATH, os.pardir))
DEBUG_FOLDER = os.path.join(SCRIPT_FOLDER, 'cleanimg_DEBUG')
WORK_FOLDER = os.path.join(SCRIPT_FOLDER, 'cleanimg_WORK')


class Colour(object):

	def __str__( self ):
		return self.name
	
	def __eq__( self, other ):
		# print('comparing ' + str(self) + ' and ' + str(other))
		if type(other) == str:
			try:
				other = Colour(other)
			except:
				return False
		if not isinstance(other, Colour):
			return False
		return (self.red == other.red) and (self.green == other.green) and (self.blue == other.blue) and (self.alpha == other.alpha)
	
	@staticmethod
	def rgb( red: int, green: int, blue: int ):
		colour = Colour()
		colour.name = 'rgb(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ')'
		if red < 0 or red > 255:
			raise ValueError('red not in range [0..255]: ' + str(red))
		if green < 0 or green > 255:
			raise ValueError('green not in range [0..255]: ' + str(red))
		if blue < 0 or blue > 255:
			raise ValueError('blue not in range [0..255]: ' + str(red))			
		colour.red = red
		colour.green = green
		colour.blue = blue
		colour.alpha = 1
		return colour
		
	@staticmethod
	def rgba( red: int, green: int, blue: int, alpha: float ):
		colour = Colour()
		colour.name = 'rgba(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ', ' + str(alpha) + ')'
		if red < 0 or red > 255:
			raise ValueError('red not in range [0..255]: ' + str(red))
		if green < 0 or green > 255:
			raise ValueError('green not in range [0..255]: ' + str(red))
		if blue < 0 or blue > 255:
			raise ValueError('blue not in range [0..255]: ' + str(red))
		if alpha < 0 or alpha > 1:
			raise ValueError('alpha not in range [0..1]: ' + str(alpha))
		colour.red = red
		colour.green = green
		colour.blue = blue
		colour.alpha = alpha
		return colour
	
	@staticmethod
	def hex( hexstring ):
		colour = Colour()
		full_hexstring = hexstring
		short_form = re.search('^#?([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])$', hexstring)
		if short_form:
			full_hexstring = '#' + short_form[0][0] + short_form[0][0] + short_form[0][1] + short_form[0][1] + short_form[0][2] + short_form[0][2]
		match = re.search('^#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$', full_hexstring)
		return colour
	
	def copy(self):
		colour = Colour()
		colour.name = self.name
		colour.red = self.red
		colour.green = self.green
		colour.blue = self.blue
		colour.alpha = self.alpha
		return colour
	
	def getAverageBrightness(self):
		brightness = (0.299 * self.red) + (0.587 * self.green) + (0.114 * self.blue)
		return brightness
	
	def toTuple( self ):
		return ( self.red, self.green, self.blue, alphaFloatToInt255(self.alpha) )

	def __init__( self, text: str = None ):
	
		if text is None:
			return
		
		if type(text) == tuple or type(text) == list and len(text) >= 3:
			red = int(text[0])
			green = int(text[1])
			blue = int(text[2])
			self.name = 'rgb(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ')'
			if red < 0 or red > 255:
				raise ValueError('red not in range [0..255]: ' + str(red))
			if green < 0 or green > 255:
				raise ValueError('green not in range [0..255]: ' + str(red))
			if blue < 0 or blue > 255:
				raise ValueError('blue not in range [0..255]: ' + str(red))
			self.red = red
			self.green = green
			self.blue = blue
			if len(text) >= 4:
				alpha = text[3]
				if alpha < 0 or alpha > 255:
					raise ValueError('alpha not in range [0..255]: ' + str(alpha))
				else:
					self.alpha = alphaInt255ToFloat(alpha)
			else:
				self.alpha = 1
			return
		
		rgbMatch = re.findall(r'^rgb\( ?([0-9]+), ?([0-9]+), ?([0-9]+) ?\)$', text)
		rgbaMatch = re.findall(r'^rgba\( ?([0-9]+), ?([0-9]+), ?([0-9]+), ?([0-9]+) ?\)$', text)
		if rgbMatch or rgbaMatch:
		
			if rgbMatch:
				red = int(rgbMatch[0][0])
				green = int(rgbMatch[0][1])
				blue = int(rgbMatch[0][2])
				alpha = 1
				self.name = 'rgb(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ')'
			elif rgbaMatch:
				red = int(rgbaMatch[0][0])
				green = int(rgbaMatch[0][1])
				blue = int(rgbaMatch[0][2])
				alpha = float(rgbaMatch[0][3])
				self.name = 'rgba(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ', ' + str(alpha) + ')'
				
			if red < 0 or red > 255:
				raise ValueError('red not in range [0..255]: ' + str(red))
			if green < 0 or green > 255:
				raise ValueError('green not in range [0..255]: ' + str(red))
			if blue < 0 or blue > 255:
				raise ValueError('blue not in range [0..255]: ' + str(red))
			if alpha < 0 or alpha > 1:
				raise ValueError('alpha not in range [0..1]: ' + str(red))
			self.red = red
			self.green = green
			self.blue = blue
			self.alpha = alpha
			return
		
		hex3Match = re.findall(r'^#?([A-Fa-f0-9])([A-Fa-f0-9])([A-Fa-f0-9])$', text)
		hex6Match = re.findall(r'^#?([A-Fa-f0-9]{2})([A-Fa-f0-9]{2})([A-Fa-f0-9]{2})$', text)
		if hex3Match or hex6Match:
		
			if hex3Match:
				redString = hex3Match[0][0] + hex3Match[0][0]
				greenString = hex3Match[0][1] + hex3Match[0][1]
				blueString = hex3Match[0][2] + hex3Match[0][2]
			elif hex6Match:
				redString = hex6Match[0][0]
				greenString = hex6Match[0][1]
				blueString = hex6Match[0][2]
		
			self.name = '#' + redString.upper() + greenString.upper() + blueString.upper()
			
			self.red = int(redString, 16)
			self.green = int(greenString, 16)
			self.blue = int(blueString, 16)
			
			self.alpha = 1
			return
		
		self.name = None
		
		if text == 'white':
			self.name = 'white';
			self.red = 255
			self.green = 255
			self.blue = 255
			self.alpha = 1
		elif text == 'black':
			self.name = 'black';
			self.red = 0
			self.green = 0
			self.blue = 0
			self.alpha = 1
		elif text == 'gray' or text == 'grey':
			self.name = text;
			self.red = 128
			self.green = 128
			self.blue = 128
			self.alpha = 1
		elif text == 'red':
			self.name = 'red';
			self.red = 255
			self.green = 0
			self.blue = 0
			self.alpha = 1
		elif text == 'green':
			self.name = 'green';
			self.red = 0
			self.green = 255
			self.blue = 0
			self.alpha = 1
		elif text == 'blue':
			self.name = 'blue';
			self.red = 0
			self.green = 0
			self.blue = 255
			self.alpha = 1
		elif text == 'cyan':
			self.name = 'cyan';
			self.red = 0
			self.green = 255
			self.blue = 255
			self.alpha = 1
		elif text == 'magenta':
			self.name = 'magenta';
			self.red = 255
			self.green = 255
			self.blue = 0
			self.alpha = 1
		elif text == 'transparent':
			self.name = 'transparent';
			self.red = 0
			self.green = 0
			self.blue = 0
			self.alpha = 0
	
		if self.name is None:
			raise ValueError(text)
	
	def getChromaDistance(self, other):
		if type(other) == str:
			try:
				other = Colour(other)
			except:
				return 10000000000
		if not isinstance(other, Colour):
			return 10000000000
		
		distance_r = (self.red - other.red)
		distance_g = (self.green - other.green)
		distance_b = (self.blue - other.blue)
		
		chroma_distance = distance_r*distance_r + distance_g*distance_g + distance_b*distance_b
		
		return chroma_distance

def union( *types ):
	def output( text ):
		for t in types:
			primitive = False
			if t is None:
				primitive = True
				if text == 'None' or text == 'none' or text is None:
					return None
			else:
				for p in [int, str, bool]:
					if isinstance(t, p):
						primitive = True
						try:
							if p(text) == t:
								return p(text)
							break
						except:
							pass
			if not primitive:
				try:
					return t(text)
				except:
					pass
		raise ValueError(text)
	return output

parser = argparse.ArgumentParser(description='A horrifically inefficient but hopefully effective way to remove a background from an image')
parser.add_argument('-c', '--colour', '--color', type=union(Colour, 'infer'), default='infer', help='specifies the background colour to remove. if omitted, will be inferred from the most prevalent colour around the image edge')
parser.add_argument('-g', '--guide', type=union(Colour, None), default='none', help='specifies the guide colour. pixels of the guide colour are assumed to be transparent, and exploration will begin around the guide pixels. when there are blocks of the same colour of the background which shouldn\'t be become transparent')
parser.add_argument('-b', '--border', type=union(Colour, None), default='none', help='specifies the colour of the border, allowing for more precise calculation of a pixel\'s original colour and transparency, but less useful when there are multiple borders.')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='produces additional text in the console showing the current state of the program')
parser.add_argument('-a', '--artefact', '--artifact', '--min', type=float, default=0.05, help='sets the minimum alpha threshold. expects a value between 0 and 1 inclusive. a pixel with calculated opacity less than or equal to the threshold will be treated as fully transparent. useful for removing jpeg artifacts')
parser.add_argument('-t', '--threshold', '--max', type=float, default=0.8, help='sets the maximum alpha threshold. expects a value between 0 and 1 inclusive. a pixel with calculated opacity greater than or equal to the threshold will not allow exploration to neighbouring pixels. defaults to 0.8')
parser.add_argument('-o', '--overwrite', action='store_true', default=False, help='if this flag is present, the program will not prompt to overwrite the output file if it already exists, and will instead silently overwrite it')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='enables debugging mode. every frame of the iteration process will be logged, with red pixels showing the current pixels being explored')
parser.add_argument('-r', '--resume', type=str, default='', help='supplies an id for this invocation, and makes the conversion resumable. Intermediate frame images will be saved to the work directory. Supplying the same resume id as a previously aborted resumable invocation will cause it to resume from that frame')
parser.add_argument('-p', '--optimise', '--optimize', action='store_true', default=False, help='optimizes animations by merging identical frames and combining their durations, resulting in a smaller image')
parser.add_argument('inputfile', type=str, nargs=1)
parser.add_argument('outputfile', type=str, nargs=1)

args = parser.parse_args()

DEBUG = args.debug
VERBOSE = args.verbose
if VERBOSE:
	print()
	print('cleanimg arguments')
	print(' - inputfile: ' + args.inputfile[0])
	print(' - outputfile: ' + args.outputfile[0])
	print(' - background colour: ' + str(args.colour))
	print(' - border colour: ' + str(args.border))
	print(' - guide colour: ' + str(args.guide))
	print(' - min alpha threshold: ' + str(args.artefact))
	print(' - max alpha threshold: ' + str(args.threshold))
	print(' - resume: ' + str(args.resume))
	print(' - optimise: ' + str(args.optimise))
	print(' - overwrite: ' + str(args.overwrite))
	print(' - verbose: ' + str(args.verbose))
	print(' - debug: ' + str(args.debug))
	print()
	
guideColour = args.guide
backgroundColour = args.colour
borderColour = args.border
minThreshold = args.artefact
maxThreshold = args.threshold
resumeId = args.resume
optimize = args.optimise
preserve = bool(args.resume)

if resumeId:
	SCRIPTID = resumeId;

INPUT_FILE_PATH = None
if args.inputfile[0].startswith('/'):
	INPUT_FILE_PATH = args.inputfile[0]
else:
	INPUT_FILE_PATH = os.path.join(CURRENT_FOLDER, args.inputfile[0])
	
OUTPUT_FILE_PATH = None
if args.outputfile[0].startswith('/'):
	OUTPUT_FILE_PATH = args.outputfile[0]
else:
	OUTPUT_FILE_PATH = os.path.join(CURRENT_FOLDER, args.outputfile[0])

if INPUT_FILE_PATH is None or not os.path.exists(INPUT_FILE_PATH):
	print('inputfile "' + INPUT_FILE_PATH + '" does not exist')
	sys.exit(1)
	
if OUTPUT_FILE_PATH is None:
	print('this should never happen')
	sys.exit(1)
	
if (not args.overwrite) and os.path.exists(OUTPUT_FILE_PATH):
	choice = input('outputfile "' + OUTPUT_FILE_PATH + '" exists. overwrite? Y/N: ')
	if choice.lower().strip() != 'y' and choice.lower().strip() != 'yes':
		sys.exit(1)

if not os.path.exists(WORK_FOLDER):
	os.mkdir(WORK_FOLDER)
if DEBUG and not os.path.exists(DEBUG_FOLDER):
	os.mkdir(DEBUG_FOLDER)

def saveImageSequence( path, image_sequence ):
	image_sequence[0].save(path, save_all=True, append_images=image_sequence[1:], loop=0)


def inferBackgroundColour( image ):

	edgePixels = { }
	
	for x in range(image.width):
	
		array = image.getpixel((x,0))
		colour = Colour.rgb(array[0], array[1], array[2])
		edgePixels[colour.name] = edgePixels.get(colour.name, 0) + 1
		
		array = image.getpixel((x,image.height-1))
		colour = Colour.rgb(array[0], array[1], array[2])
		edgePixels[colour.name] = edgePixels.get(colour.name, 0) + 1
		
	for y in range(image.height):
	
		array = image.getpixel((1,y))
		colour = Colour.rgb(array[0], array[1], array[2])
		edgePixels[colour.name] = edgePixels.get(colour.name, 0) + 1
		
		array = image.getpixel((image.width-2,y))
		colour = Colour.rgb(array[0], array[1], array[2])
		edgePixels[colour.name] = edgePixels.get(colour.name, 0) + 1
	
	inferredColour = None
	
	mostCommonColourCount = max(edgePixels.values())
	for colour in edgePixels:
		if edgePixels[colour] == mostCommonColourCount:
			inferredColour = Colour(colour)
			break
	
	return inferredColour


TRANSPARENT = Colour('transparent') # Colour.rgba(255,0,0,0.5)


VERBOSE_PIXEL_ERROR_MAP = { }

def inferOriginalColourAndTransparency( pixelColour, backgroundColour, borderColour, minThreshold, maxThreshold ):
	if pixelColour == TRANSPARENT or pixelColour == backgroundColour:
		return TRANSPARENT.copy()
	
	if pixelColour.alpha < 1:
		# it's partially transparent
		return pixelColour
	
	if borderColour:
		# pixelColour = (1 - alpha) * backgroundColour + (alpha * borderColour)
		# pixelColour = backgroundColour - (backgroundColour * alpha) + (alpha * borderColour)
		# pixelColour - backgroundColour = (alpha * borderColour) - (backgroundColour * alpha)
		# pixelColour - backgroundColour = alpha * (borderColour - backgroundColour)
		# (pixelColour - backgroundColour) / (borderColour - backgroundColour) = alpha
		
		alpha_red = None
		alpha_green = None
		alpha_blue = None
		
		total = 0
		divisor = 0
		
		max_difference = 0.25
		
		if borderColour.red != backgroundColour.red:
			alpha_red = (pixelColour.red - backgroundColour.red) / (1.0 * (borderColour.red - backgroundColour.red))
			total += alpha_red
			divisor += 1
		if borderColour.green != backgroundColour.green:
			alpha_green = (pixelColour.green - backgroundColour.green) / (1.0 * (borderColour.green - backgroundColour.green))
			total += alpha_green
			divisor += 1
		if borderColour.blue != backgroundColour.blue:
			alpha_blue = (pixelColour.blue - backgroundColour.blue) / (1.0 * (borderColour.blue - backgroundColour.blue))
			total += alpha_blue
			divisor += 1
		
		if math.fabs(alpha_red - alpha_green) > max_difference or math.fabs(alpha_red - alpha_blue) > max_difference or math.fabs(alpha_green - alpha_blue) > max_difference:
			# alpha values are too different, likely not dealing with borderColour
			return pixelColour
		
		if divisor > 0:
			average = total / divisor
		else:
			# no divisor = border colour is perfect match for background colour
			return pixelColour
		
		if average < 0 and average > -0.1:
			average = 0
			
		if average > 1 and average < 1.1:
			average = 1
		
		if average >= 0 and average <= 1:
			newPixelColour = Colour.rgba(borderColour.red, borderColour.green, borderColour.blue, average)
			return newPixelColour
		else:
			errorName = (str(pixelColour)+'_'+str(backgroundColour))
			if VERBOSE and not (errorName in VERBOSE_PIXEL_ERROR_MAP):
				print(str(pixelColour) + ' and backgroundColour ' + str(backgroundColour) + ' resulted in calculated alpha of ' + str(average))
				VERBOSE_PIXEL_ERROR_MAP[errorName] = pixelColour
			return pixelColour
	
	#if VERBOSE:
	#	print(' - - - guesstimating border colour and opacity for pixel ' + str(pixelColour) + ' with background colour ' + str(backgroundColour))
	
	minDeltaN = None
	minDelta = 255 * 255 * 3
	minDeltaColour = None
	
	leastSimilarColour = None
	leastSimilarColourDistance = None
	leastSimilarColourOpacity = None
	
	for x in range(256):
		n = x / 255
		calculatedBorderColour = calculateBorderColour(pixelColour, backgroundColour, n)
		if calculatedBorderColour is None:
			continue
		
		differenceFromBackground = calculatedBorderColour.getChromaDistance(backgroundColour)
		
		if leastSimilarColour is None:
			leastSimilarColour = calculatedBorderColour
			leastSimilarColourDistance = differenceFromBackground
			leastSimilarColourOpacity = n
			continue
		
		if differenceFromBackground > leastSimilarColourDistance:
			leastSimilarColour = calculatedBorderColour
			leastSimilarColourDistance = differenceFromBackground
			leastSimilarColourOpacity = n
			continue
		
		#if VERBOSE:
		#	print('- - - - calculated colour for opacity ' + str(n) + ' = ' + str(calculatedBorderColour))
		
		'''
		calculated_delta_red = pixelColour.red - (calculatedBorderColour.red * n + backgroundColour.red * (1 - n))
		calculated_delta_green = pixelColour.green - (calculatedBorderColour.green * n + backgroundColour.green * (1 - n))
		calculated_delta_blue = pixelColour.blue - (calculatedBorderColour.blue * n + backgroundColour.blue * (1 - n))
	
		calculated_delta = (calculated_delta_red * calculated_delta_red) + (calculated_delta_green * calculated_delta_green) + (calculated_delta_blue * calculated_delta_blue)
		
		bgColourDeltaR2 = (calculatedBorderColour.red - backgroundColour.red)*(calculatedBorderColour.red - backgroundColour.red)
		bgColourDeltaG2 = (calculatedBorderColour.green - backgroundColour.green)*(calculatedBorderColour.green - backgroundColour.green)
		bgColourDeltaB2 = (calculatedBorderColour.blue - backgroundColour.blue)*(calculatedBorderColour.blue - backgroundColour.blue)
		
		bgColourDelta2 = bgColourDeltaR2 + bgColourDeltaG2 + bgColourDeltaB2
		if bgColourDelta2 < 2500:
			continue
		
		if calculated_delta < minDelta:
			minDeltaColour = calculated_delta
			minDeltaColour = calculatedBorderColour
			minDeltaN = n
		'''
	
	if VERBOSE:
		#print(' - - - best guesstimatate[1] = opacity ' + str(minDeltaN) + ' with border colour ' + str(minDeltaColour))
		#print(' - - - best guesstimatate[2] = opacity ' + str(leastSimilarColourOpacity) + ' with border colour ' + str(leastSimilarColour) + ' ... distance = ' + str(leastSimilarColourDistance))
		pass
		
	#if minDeltaN < minThreshold:
	#	return Colour('transparent')
	#if minDeltaN <= maxThreshold and (minDeltaColour is not None):
	#	return Colour.rgba(minDeltaColour.red, minDeltaColour.green, minDeltaColour.blue, minDeltaN)
	
	if leastSimilarColourOpacity < minThreshold:
		return Colour('transparent')
	if leastSimilarColourOpacity <= maxThreshold and (leastSimilarColour is not None):
		return Colour.rgba(leastSimilarColour.red, leastSimilarColour.green, leastSimilarColour.blue, leastSimilarColourOpacity)
		
	# pixel exceeds max threshold, and so should be fully opaque
	# print("pixelColour %s exceeds threshold and will be treated as opaque" % pixelColour)
	
	return pixelColour

def calculateBorderColour( pixelColour, backgroundColour, n ):
	if n == 0:
		return pixelColour
	if n == 1:
		return backgroundColour
	borderColour_red = ((pixelColour.red - backgroundColour.red) / n) + backgroundColour.red
	borderColour_green = ((pixelColour.green - backgroundColour.green) / n) + backgroundColour.green
	borderColour_blue = ((pixelColour.blue - backgroundColour.blue) / n) + backgroundColour.blue
	
	if borderColour_red < 0 and borderColour_red > - 10:
		borderColour_red = 0
	if borderColour_green < 0 and borderColour_green > - 10:
		borderColour_green = 0
	if borderColour_blue < 0 and borderColour_blue > - 10:
		borderColour_blue = 0
		
	if borderColour_red > 255 and borderColour_red < 265:
		borderColour_red = 255
	if borderColour_green > 255 and borderColour_green < 265:
		borderColour_green = 255
	if borderColour_blue > 255 and borderColour_blue < 265:
		borderColour_blue = 255
	
	if borderColour_red < 0 or borderColour_green < 0 or borderColour_blue < 0:
		return None
	if borderColour_red > 255 or borderColour_green > 255 or borderColour_blue > 255:
		return None
	
	return Colour.rgb(int(borderColour_red), int(borderColour_green), int(borderColour_blue))

def alphaFloatToInt255( alphaFloat ):
	return int(alphaFloat * 255)
	
def alphaInt255ToFloat( alphaInt255 ):
	return (alphaInt255 / 255.0)

def __cleanimgIterative( image, backgroundColour, borderColour, minThreshold, maxThreshold, pixels, checkedPixels=None, imageNumber=1, iterationCount=1 ):
	
	if VERBOSE:
		print(' - - iteration ' + str(iterationCount) + ' (' + str(len(pixels)) + ' pixels)')
	
	if DEBUG:
		copy = image.copy()
		for pixel in pixels:
			copy.putpixel(pixel, (255, 0, 0, 255))
		copyPath = os.path.join(DEBUG_FOLDER, SCRIPTID + '_frame_' + str(imageNumber) + '_iteration_' + str(iterationCount) + '.png')
		copy.save(copyPath)
	
	if len(pixels) == 0:
		return
	if checkedPixels is None:
		checkedPixels = { }
		
	finished = True
	nextPixels = { }
	for pixel in pixels:
	
		checkedPixels[pixel] = True
		originalPixelColour = Colour(image.getpixel(pixel))
		newPixelColour = inferOriginalColourAndTransparency(originalPixelColour, backgroundColour, borderColour, minThreshold, maxThreshold)
		if originalPixelColour != newPixelColour:
			image.putpixel(pixel, newPixelColour.toTuple())
		
		# print(str(pixel) + ': ' + str(originalPixelColour) + ' => ' + str(newPixelColour))
		if newPixelColour.alpha < maxThreshold:
			adjacentPixels = [ ]
			if pixel[0] > 0:
				adjacentPixels += [(pixel[0]-1, pixel[1])] # Left
			if pixel[1] > 0:
				adjacentPixels += [(pixel[0], pixel[1]-1)] # Above
			if pixel[0] < image.width-1:
				adjacentPixels += [(pixel[0]+1, pixel[1])] # Right
			if pixel[1] < image.height-1:
				adjacentPixels += [(pixel[0], pixel[1]+1)] # Below
		
			for adjacentPixel in adjacentPixels:
				nextPixels[adjacentPixel] = True
				
			# print('adjacent pixels added: ' + str(adjacentPixels))
	
	for nextPixel in list(nextPixels.keys()):
		if nextPixel in checkedPixels:
			del nextPixels[nextPixel]
	
	__cleanimgIterative(image, backgroundColour, borderColour, minThreshold, maxThreshold, nextPixels, checkedPixels, imageNumber, iterationCount+1)


def cleanimg( image, backgroundColour=None, guideColour=None, borderColour=None, minThreshold=0, maxThreshold=1, imageNumber=1 ):
	
	inferred = False
	if backgroundColour == 'infer':
		backgroundColour = inferBackgroundColour(image)
		inferred = True
		if VERBOSE:
			print(' - inferred background colour: ' + str(backgroundColour))
	
	if backgroundColour is None:
		raise Exception('unable to determine background colour')
	
	initialPixels = { }
	
	for x in range(0, image.width):
		for y in range(0, image.height):
			pixelColour = None
			if inferred and guideColour is None:
				# we only care about edge pixels
				if x == 0 or y == 0 or x == image.width-1 or y == image.height-1:
					pixelColour = image.getpixel((x,y))
				else:
					continue
			else:
				pixelColour = image.getpixel((x,y))
			
			#print('(%d,%d): %s' % (x, y, Colour(pixelColour)))
			
			if guideColour:
				if Colour(pixelColour) == guideColour:
					initialPixels[(x, y)] = True
			elif Colour(pixelColour) == backgroundColour:
					initialPixels[(x, y)] = True
	
	if VERBOSE:
			print(' - initial pixels: ' + str(len(initialPixels)))
	
	# treat guide pixels as perfect matches for background colour
	cleanedImage = image.copy()
	
	if guideColour:
		for pixelCoord in initialPixels:
			cleanedImage.putpixel(pixelCoord, backgroundColour.toTuple())
	
	emptyDict = { }
	
	__cleanimgIterative(cleanedImage, backgroundColour, borderColour, minThreshold, maxThreshold, initialPixels, emptyDict, imageNumber)
	
	return cleanedImage;
	

inputImage = Image.open(INPUT_FILE_PATH)

animated = False
frameCount = 0
try:
	if inputImage.is_animated:
		frameCount = inputImage.n_frames
		animated = True
except AttributeError:
	frameCount = 0

if VERBOSE:
	print()
	print('input image details')
	print(' - format: ' + str(inputImage.format))
	print(' - size: ' + str(inputImage.size))
	print(' - animated: ' + str(animated) + ((' (' + str(frameCount) + ' frames)') if frameCount else ''))

durations = [ ]
inputImages = [ ]
if animated:
	for n in range(0, frameCount):
		inputImage.seek(n)
		durations += [inputImage.info['duration']]
		frameImage = inputImage.convert("RGBA")
		inputImages += [frameImage]
else:
	inputImages += [inputImage.convert("RGBA")]

'''
if backgroundColour == 'infer' and len(inputImages) > 1:
	if VERBOSE:
		print(' - inferring background across frames')

	imageInferredBackgroundColours = [ ]
	n = 0
	for image in inputImages:
		n += 0
		imageInferredBackgroundColours += [inferBackgroundColour(image)]
		if VERBOSE:
			print(' - - image ' + str(image) + ' inferred background colour: ' + str(backgroundColour))
	
	if VERBOSE:
		print(' - inferred background colour: ' + str(imageInferredBackgroundColours[0]))
'''

def compareImages( a, b ):
	#if a == b:
	#	return True
	if a.width != b.width or a.height != b.height:
		if VERBOSE:
			print('- images are different heights or widths')
		return False
	for x in range(a.width):
		for y in range(a.height):
			a_colour = a.getpixel((x,y))
			b_colour = b.getpixel((x,y))
			if a_colour[3] == 0 and b_colour[3] == 0:
				continue
			if a_colour[0] == b_colour[0] and a_colour[1] == b_colour[1] and a_colour[2] == b_colour[2] and a_colour[3] == b_colour[3]:
				continue
			if VERBOSE:
				print('- pixels at (' + str(x) + ',' + str(y) + ') are different colours: ' + str(a_colour) + ', ' + str(b_colour))
			return False
	# all pixels are identical
	if VERBOSE:
		print('- images are identical')
	return True

cleanedImages = [ ]
cleanedImagePaths = [ ]
n = 0
for image in inputImages:
	n += 1
	if VERBOSE:
		print()
		print('image ' + str(n) + ((' (duration=' +str(durations[n-1]) + 'ms)') if animated else ''))
	if resumeId:
		cleanedImagePath = os.path.join(WORK_FOLDER, SCRIPTID + '_frame_' + str(n) + '.png')
		if os.path.exists(cleanedImagePath):
			try:
				cleanedImage = Image.open(cleanedImagePath).convert("RGBA")
				cleanedImages += [cleanedImage]
				if VERBOSE:
					print('image ' + str(n) + ' found in work directory from last run at ' + cleanedImagePath)
				cleanedImagePaths += [cleanedImagePath]
				continue;
			except:
				pass
	cleanedImage = cleanimg( image, backgroundColour, guideColour, borderColour, minThreshold, maxThreshold, n )
	if resumeId:
		cleanedImagePath = os.path.join(WORK_FOLDER, SCRIPTID + '_frame_' + str(n) + '.png')
		if VERBOSE:
			print('- saving work image to file: ' + cleanedImagePath)
		cleanedImage.save(cleanedImagePath)
		cleanedImagePaths += [cleanedImagePath]
		# save as we go
	cleanedImages += [cleanedImage]


cleanedImagesToKeep = [ ]
cleanedImagesToKeepDurations = [ ]
cleanedImagePathsToKeep = [ ]

if len(cleanedImages) > 1 and optimize:
	if VERBOSE:
		print()
	i = 0
	for i in range(len(cleanedImages)):
		cleanedImage = cleanedImages[i]
		duration = durations[i]
		if i > 1 and VERBOSE:
			print('comparing frame ' + str(i-1) + ' image with frame ' + str(i) + ' image')
		if len(cleanedImagesToKeep) == 0:
			cleanedImagesToKeep.append(cleanedImage)
			cleanedImagesToKeepDurations.append(duration)
			if resumeId:
				cleanedImagePath = cleanedImagePaths[i]
				cleanedImagePathsToKeep.append(cleanedImagePath)
		elif compareImages(cleanedImagesToKeep[-1],cleanedImage):
			cleanedImagesToKeepDurations[-1] += duration
		else:
			cleanedImagesToKeep.append(cleanedImage)
			cleanedImagesToKeepDurations.append(duration)
			if resumeId:
				cleanedImagePath = cleanedImagePaths[i]
				cleanedImagePathsToKeep.append(cleanedImagePath)
else:
	cleanedImagesToKeep = cleanedImages
	cleanedImagesToKeepDurations = durations
	cleanedImagePathsToKeep = cleanedImagePaths

if len(cleanedImagesToKeep) > 1 and (OUTPUT_FILE_PATH.lower().endswith('.png') or OUTPUT_FILE_PATH.lower().endswith('.apng')):

	if VERBOSE and not DEBUG:
		print()
		print('generating work for combination')
	
	if not resumeId:
		n = 0
		for cleanedImage in cleanedImagesToKeep:
			n += 1
			cleanedImagePath = os.path.join(WORK_FOLDER, SCRIPTID + '_frame_' + str(n) + '.png')
			if VERBOSE:
				print('saving work image to file: ' + cleanedImagePath)
			cleanedImage.save(cleanedImagePath)
			cleanedImagePaths += [cleanedImagePath]
			cleanedImagePathsToKeep += [cleanedImagePath]
	
	if VERBOSE:
		if DEBUG:
			print()
		print('generating output image (apng) - with ' + str(len(cleanedImagesToKeep)) + ' frames')
	
	outputApng = APNG()
	for i in range(len(cleanedImagePathsToKeep)):
		cleanedImagePath = cleanedImagePathsToKeep[i]
		duration = cleanedImagesToKeepDurations[i]
		if VERBOSE:
			print('- (' + str(duration) + ' ms) ' + cleanedImagePath)
		outputApng.append_file(cleanedImagePath, delay=duration)
	outputApng.save(OUTPUT_FILE_PATH)
	
	if VERBOSE and (not DEBUG) and (not preserve):
		print('cleaning up work images')
	
	if not DEBUG and (not preserve):
		for cleanedImagePath in cleanedImagePaths:
			os.remove(cleanedImagePath)
else:
	if VERBOSE and not DEBUG:
		print('generating output image')
	saveImageSequence(OUTPUT_FILE_PATH, cleanedImages)



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
		
		rgbMatch = re.findall('^rgb\( ?([0-9]+), ?([0-9]+), ([0-9]+) ?\)$', text)
		if rgbMatch:
			red = int(rgbMatch[0][0])
			green = int(rgbMatch[0][1])
			blue = int(rgbMatch[0][2])
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
			self.alpha = 1
			return
		
		# TODO
		
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
		elif text == 'transparent':
			self.name = 'transparent';
			self.red = 0
			self.green = 0
			self.blue = 0
			self.alpha = 0
	
		if self.name is None:
			raise ValueError(text)

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
parser.add_argument('-t', '--threshold', type=float, default=0.8, help='sets the alpha threshold. expects a value between 0 and 1 inclusive. a pixel with calculated opacity greater than or equal to the threshold will not allow exploration to neighbouring pixels. defaults to 0.8')
parser.add_argument('-o', '--overwrite', action='store_true', default=False, help='if this flag is present, the program will not prompt to overwrite the output file if it already exists, and will instead silently overwrite it')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='enables debugging mode. every frame of the iteration process will be logged, with red pixels showing the current pixels being explored')
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
	print(' - alpha threshold: ' + str(args.threshold))
	print(' - overwrite: ' + str(args.overwrite))
	print(' - verbose: ' + str(args.verbose))
	print(' - debug: ' + str(args.debug))
	print()
	
guideColour = args.guide
backgroundColour = args.colour
borderColour = args.border
threshold = args.threshold

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
	choice = input('outputfile "' + INPUT_FILE_PATH + '" exists. overwrite? Y/N: ')
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

def inferOriginalColourAndTransparency( pixelColour, backgroundColour, borderColour, threshold ):
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
		
		if average >= 0:
			newPixelColour = Colour.rgba(borderColour.red, borderColour.green, borderColour.blue, average)
			return newPixelColour
		else:
			errorName = (str(pixelColour)+'_'+str(backgroundColour))
			if VERBOSE and not (errorName in VERBOSE_PIXEL_ERROR_MAP):
				print(str(pixelColour) + ' and backgroundColour ' + str(backgroundColour) + ' resulted in calculated alpha of ' + str(average))
				VERBOSE_PIXEL_ERROR_MAP[errorName] = pixelColour
			return pixelColour
		
	# TODO
	return pixelColour

def alphaFloatToInt255( alphaFloat ):
	return int(alphaFloat * 255)
	
def alphaInt255ToFloat( alphaInt255 ):
	return (alphaInt255 / 255.0)

def __cleanimgIterative( image, backgroundColour, borderColour, threshold, pixels, checkedPixels=None, imageNumber=1, iterationCount=1 ):
	
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
		newPixelColour = inferOriginalColourAndTransparency(originalPixelColour, backgroundColour, borderColour, threshold)
		if originalPixelColour != newPixelColour:
			image.putpixel(pixel, newPixelColour.toTuple())
		
		# print(str(pixel) + ': ' + str(originalPixelColour) + ' => ' + str(newPixelColour))
		if newPixelColour.alpha < threshold:
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
	
	__cleanimgIterative(image, backgroundColour, borderColour, threshold, nextPixels, checkedPixels, imageNumber, iterationCount+1)


def cleanimg( image, backgroundColour=None, guideColour=None, borderColour=None, threshold=1, imageNumber=1 ):
	
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
			cleanedImage.putpixel(pixelCoord, backgroundColour)
	
	emptyDict = { }
	
	__cleanimgIterative(cleanedImage, backgroundColour, borderColour, threshold, initialPixels, emptyDict, imageNumber)
	
	return cleanedImage;
	

inputImage = Image.open(INPUT_FILE_PATH)

frameCount = 0
if inputImage.is_animated:
	frameCount = inputImage.n_frames

if VERBOSE:
	print()
	print('input image details')
	print(' - format: ' + str(inputImage.format))
	print(' - size: ' + str(inputImage.size))
	print(' - animated: ' + str(inputImage.is_animated) + ((' (' + str(frameCount) + ' frames)') if frameCount else ''))

durations = [ ]
inputImages = [ ]
if inputImage.is_animated:
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

cleanedImages = [ ]
n = 0
for image in inputImages:
	n += 1
	if VERBOSE:
		print()
		print('image ' + str(n) + ' (duration=' +str(durations[n-1]) + 'ms)')
	cleanedImage = cleanimg( image, backgroundColour, guideColour, borderColour, threshold, n )
	cleanedImages += [cleanedImage]


if len(cleanedImages) > 1 and (OUTPUT_FILE_PATH.lower().endswith('.png') or OUTPUT_FILE_PATH.lower().endswith('.apng')):

	if VERBOSE and not DEBUG:
		print()
		print('generating work for combination')
		
	cleanedImagePaths = [ ]
	n = 0
	for cleanedImage in cleanedImages:
		n += 1
		cleanedImagePath = os.path.join(WORK_FOLDER, SCRIPTID + '_frame_' + str(n) + '.png')
		cleanedImage.save(cleanedImagePath)
		cleanedImagePaths += [cleanedImagePath]
	
	if VERBOSE:
		if DEBUG:
			print()
		print('generating output image (apng)')
		
	outputApng = APNG()
	for i in range(len(cleanedImagePaths)):
		cleanedImagePath = cleanedImagePaths[i]
		duration = durations[i]
		outputApng.append_file(cleanedImagePath, delay=duration)
	outputApng.save(OUTPUT_FILE_PATH)
	
	if VERBOSE and not DEBUG:
		print('cleaning up work images')
	
	if not DEBUG:
		for cleanedImagePath in cleanedImagePaths:
			os.remove(cleanedImagePath)
else:
	if VERBOSE and not DEBUG:
		print('generating output image')
	saveImageSequence(OUTPUT_FILE_PATH, cleanedImages)


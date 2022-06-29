
# TL Note: It's pronounced GIF, not GIF

import os
import re
import sys
import argparse
from PIL import Image

class Colour(object):

	def __str__( self ):
		return self.name
	
	def __eq__( self, other ):
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
		colour.name = 'rgb(' + str(red) + ', ' + str(green) + ', ' + str(blue) + ')'
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
	
	def copy():
		colour = Colour()
		colour.name = name
		colour.red = red
		colour.green = green
		colour.blue = blue
		colour.alpha = colour.alpha
		return colour

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

parser = argparse.ArgumentParser(description='An inefficient but hopefully effective way to remove a background from an image')
parser.add_argument('-c', '--colour', '--color', type=union(Colour, 'infer'), default='infer', help='specifies the background colour to remove')
parser.add_argument('-g', '--guide', type=union(Colour, None), default='none', help='specifies the guide colour. pixels of the guide colour are assumed to be transparent, and those adjacent to them are assumed to be part of the background')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='produces more text in the console')
parser.add_argument('-t', '--threshold', type=float, default=1, help='sets the alpha threshold. a pixel with calculated opacity greater than or equal to this will not allow exploration to neighbouring pixels')
parser.add_argument('inputfile', type=str, nargs=1)
parser.add_argument('outputfile', type=str, nargs=1)

args = parser.parse_args()

VERBOSE = args.verbose
if VERBOSE:
	print()
	print('cleanimg arguments')
	print(' - inputfile: ' + args.inputfile[0])
	print(' - outputfile: ' + args.outputfile[0])
	print(' - background colour: ' + str(args.colour))
	print(' - guide colour: ' + str(args.guide))
	print(' - alpha threshold: ' + str(args.threshold))
	print(' - verbose: ' + str(args.verbose))
	
guideColour = args.guide
backgroundColour = args.colour
threshold = args.threshold

def saveImageSequenceAsAnimatedGIF( path, image_sequence ):
	image_sequence[0].save(path, save_all=True, append_images=image_sequence[1:])


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

TRANSPARENT = Colour('transparent')

def inferOriginalColourAndTransparency( pixelColour, backgroundColour, threshold ):
	if pixelColour == TRANSPARENT or pixelColour == backgroundColour:
		return Colour('transparent')
	# TODO
	return pixelColour

def __cleanimgIterative( image, backgroundColour, threshold, pixels, checkedPixels=None ):
	
	if VERBOSE:
		print(' - - iteration')
	
	if len(pixels) == 0:
		return
	if checkedPixels is None:
		checkedPixels = { }
	
	finished = True
	nextPixels = { }
	for pixel in pixels:
		checkedPixels[pixel] = True
		pixelColour = Colour(image.getpixel(pixel))
		colour = inferOriginalColourAndTransparency(pixelColour, backgroundColour, threshold)
	
	__cleanimgIterative(image, backgroundColour, threshold, nextPixels, checkedPixels)

def cleanimg( image, backgroundColour=None, guideColour=None, threshold=1 ):
	
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
			pixel = None
			if inferred and guideColour is None:
				# we only care about edge pixels
				if x == 0 or y == 0 or x == image.width-1 or y == image.height-1:
					pixel = image.getpixel((x,y))
				else:
					continue
			else:
				pixel = image.getpixel((x,y))
			
			if guideColour:
				if Colour(pixel) == guideColour:
					initialPixels[(x, y)] = True
			elif Colour(pixel) == backgroundColour:
					initialPixels[(x, y)] = True
	
	if VERBOSE:
			print(' - initial pixels: ' + str(len(initialPixels)))
	
	# treat guide pixels as perfect matches for background colour
	cleanedImage = image.copy()
	
	if guideColour:
		for pixelCoord in initialPixels:
			cleanedImage.putpixel(pixelCoord, backgroundColour)
	
	__cleanimgIterative(cleanedImage, backgroundColour, threshold, initialPixels)
	
	return cleanedImage;
	

inputImage = Image.open('D:/Documents/GitHub/cleanimg/test.gif')

frameCount = 0
if inputImage.is_animated:
	frameCount = inputImage.n_frames

if VERBOSE:
	print()
	print('input image details')
	print(' - format: ' + str(inputImage.format))
	print(' - size: ' + str(inputImage.size))
	print(' - animated: ' + str(inputImage.is_animated) + ((' (' + str(frameCount) + ' frames)') if frameCount else ''))

inputImages = [ ]
if inputImage.is_animated:
	for n in range(0, frameCount):
		inputImage.seek(n)
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
		print('image ' + str(n))
	cleanedImage = cleanimg( image, backgroundColour, guideColour, threshold )
	cleanedImages += [cleanedImage]

n = 0
for cleanedImage in cleanedImages:
	n += 1
	cleanedImage.save('D:/Documents/GitHub/cleanimg/out_' + str(n) + '.png')

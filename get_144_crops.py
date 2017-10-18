from PIL import Image, ImageOps

# Input: Image
# Output: 144 image crops following "Going deeper with convolutions" paper

def get_144_crops(im, crop_size):

    crops = [] # Output crops
    width, height = im.size
    portait = False
    if height > width: portait = True

    rescaled_images = [] # Original image rescaled with shorter size set to 256,288,320,352
    resizing_sizes = [256,288,320,352]

    # Resize images using shorter size
    for size in resizing_sizes:
        if portait: rescaled_images.append(im.resize((size, int(float(size)/width*height)), Image.ANTIALIAS))
        else: rescaled_images.append(im.resize((int(float(size)/height*width), size), Image.ANTIALIAS))

    # For each resized image, take 3 swares (left, right, center ot top, bot, center)
    squares = []
    for i in rescaled_images:
        width, height = i.size
        if portait:
            top = (height - width) / 2
            bot = height - top
            squares.append(i.crop((0, top, width, bot)))
            squares.append(i.crop((0, 0, width, width)))
            squares.append(i.crop((0, height-width, width, height)))
        else:
            left = (width - height) / 2
            right = width - left
            squares.append(i.crop((left, 0, right, height)))
            squares.append(i.crop((0, 0, height, height)))
            squares.append(i.crop((width-height, 0, width, height)))

        # for i in squares: i.show()

    # For each square, take 4 corners and center crop and the square resized and the mirrored versions
    for s in squares:

        # Resized square
        aux = s.resize((crop_size,crop_size), Image.ANTIALIAS)
        crops.append(aux)
        crops.append(ImageOps.mirror(aux))

        # Central crop
        left = (width - crop_size) / 2
        right = width - left
        top = (height - crop_size) / 2
        bot = height - top
        aux = s.crop((left, top, right, bot))
        crops.append(aux)
        crops.append(ImageOps.mirror(aux))

        #Corner crops
        aux = s.crop((0, 0, crop_size, crop_size))
        crops.append(aux)
        crops.append(ImageOps.mirror(aux))

        width, height = s.size
        aux = s.crop((width - crop_size, height - crop_size, width, height))
        crops.append(aux)
        crops.append(ImageOps.mirror(aux))

        aux = s.crop((width - crop_size, 0, width, crop_size))
        crops.append(aux)
        crops.append(ImageOps.mirror(aux))

        aux = s.crop((0, height - crop_size, crop_size, height))
        crops.append(aux)
        crops.append(ImageOps.mirror(aux))

    # Check all crops are crop_size < crop_size
    resized = 0
    for c in range(0,len(crops)):
        if crops[c].size[0] is not crop_size or crops[c].size[1] is not crop_size:
            if crops[c].size[0] > crop_size+1 or crops[c].size[1] > crop_size+1:
                print "Warning: original crop size: " + str(crops[c].size)
            if crops[c].size[0] < crop_size-1 or crops[c].size[1] < crop_size-1:
                print "Warning: original crop size: " + str(crops[c].size)

            crops[c] = crops[c].resize((crop_size,crop_size), Image.ANTIALIAS)
            resized +=1

    return crops

# filenames = ['/home/raulgomez/Downloads/sample-l.jpg','/home/raulgomez/Downloads/sample-p.jpg']
# for f in filenames:
#     im = Image.open(f)
#     get_144_crops(im, 224)

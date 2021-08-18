from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# Opens a image in RGB mode

files = ['EventDistributionAcrosst0045GeV2.jpg',
'EventDistributionAcrosst0125GeV2.jpg',
'EventDistributionAcrosst0175GeV2.jpg',
'EventDistributionAcrosst025GeV2.jpg',
'EventDistributionAcrosst035GeV2.jpg',
'EventDistributionAcrosst050GeV2.jpg',
'EventDistributionAcrosst080GeV2.jpg',
'EventDistributionAcrosst125GeV2.jpg',
'EventDistributionAcrosst175GeV2.jpg',
'EventDistributionAcrosst250GeV2.jpg',
'EventDistributionAcrosst40GeV2.jpg',
'EventDistributionAcrosst750GeV2.jpg']

# for fname in files:
#     im = Image.open("finished_pics/{}".format(fname))

    
#     # Size of the image in pixels (size of original image)
#     # (This is not mandatory)
#     width, height = im.size
    
#     # Setting the points for cropped image
#     left = 0.01*width
#     top = 0
#     right = width
#     bottom = 0.975*2* height / 3
    
#     # Cropped image of above dimension
#     # (It will not change original image)
#     im1 = im.crop((left, top, right, bottom))
#     im1.save("cropped/{}".format(fname))
#     # Shows the image in image viewer
#     im1.close()
#     im.close()

for fname in files:
    im = Image.open("cropped/{}".format(fname))

    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    
    # Setting the points for cropped image
    # left = 0.01*width
    # top = 0
    # right = width
    # bottom = 0.975*2* height / 3
    size = int(width/10),int(height/10)
    # # Cropped image of above dimension
    # # (It will not change original image)
    # im1 = im.crop((left, top, right, bottom))
    im1 = im.resize(size, Image.ANTIALIAS)
    im1.save("lowres/{}".format(fname),dpi=(200,200))
    # Shows the image in image viewer
    im1.close()
    im.close()
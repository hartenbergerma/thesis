# import spectral as sp
# import numpy as np
# import matplotlib
# matplotlib.use("wxAgg")

# data_folder = "helicoid/004-02"

# # Load the sp data
# img = sp.open_image(data_folder + "/raw.hdr")
# white_ref = sp.open_image(data_folder + "/whiteReference.hdr")
# dark_ref = sp.open_image(data_folder + "/darkReference.hdr")
# img_gt = sp.open_image(data_folder + "/gtMap.hdr").read_band(0)

# # overlay class data on top of spectral data
# bands = tuple(map(int, img.metadata["default bands"]))
# print(type(img), type(gt_map))
# view = sp.imshow(img, bands, classes=gt_map)
# view.set_display_mode('overlay')
# view.class_alpha = 0.5

# sp.view_cube(img, bands=[00, 400, 1])

# pc = spectral.principal_components(img)
# xdata = pc.transform(img)
# w = spectral.view_nd(xdata[:,:,:15], classes=img_gt)




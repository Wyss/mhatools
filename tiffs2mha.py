import numpy as np
import tifffile as tf
import zlib
import glob
import sys

HEADER = """ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = {is_compressed}
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = {x_spacing} {y_spacing} {z_spacing}
DimSize = {image_x_resolution} {image_y_resolution} {stack_size}
ElementType = {datatype}
ElementDataFile = LOCAL
"""

def stackImages(in_path,
                in_prefix, index_digits,
                low_index=0,
                upper_index=None,
                tiff_ext='.tif'
                ):
    """ does not support negative indexing
    in_prefix must be unique for a given image stack
    """
    header_info = {}

    # 1. Get list of possible files
    glob_string = os.path.join(in_path, in_prefix + '_*[0-9]' + tiff_ext)
    possible_files = glob.glob(glob_string)

    # 2. set upper index if necessary
    if upper_index is None:
        max_seen = -1
        for fn in possible_files:
            base, ext = os.path.splitext(fn)
            filename_core = os.path.basename(base)
            _, index_str = filename_core.split(in_prefix + '_')
            index = int(index_str)
            if index > max_seen:
                max_seen = index
        upper_index = max_seen + 1
    # end if

    if upper_index < 0:
        raise IndexError(
            "negative upper_index not allowed, {}".format(upper_index))

    if low_index < 0:
        raise IndexError(
            "negative low_index not allowed, {}".format(low_index))

    if upper_index < low_index:
        raise IndexError("upper_index {} must be less than low_index {}")

    count_format = '_{:0%d}' % (index_digits)
    file_str = os.path.join(in_path, in_prefix + count_format + tiff_ext)

    # 3. validate all files in stack exist before writing any file
    for i in range(low_index, upper_index):
        the_file = file_str.format(i)
        if not os.path.isfile(the_file):
            raise ValueError("missing file {}, {} in stack".format(i, the_file))

    # 4. Open and append all tiff files to stack
    image_list = []
    shape = None
    datasize = None
    for i in range(low_index, upper_index):
        the_file = file_str.format(i)
        image = tf.imread(the_file)
        if shape is None:
            shape = image.shape
            datasize = shape[0]*shape[1]
        image_list.append(image.reshape((datasize,)))
    print("size of image_stack", len(image_list))
    datatype = image.dtype
    if image.dtype.itemsize == 2:
        datatype = 'MET_USHORT'
    elif image.dtype.itemsize == 1:
        datatype = 'MET_UCHAR'
    else:
        raise ValueError("Unsupported bit depth")
    image_x_resolution, image_y_resolution = image.shape
    header_info['image_x_resolution'] = image_x_resolution
    header_info['image_y_resolution'] = image_y_resolution
    header_info['datatype'] = datatype
    header_info['stack_size'] = upper_index - low_index

    return image_list, header_info
# end def

def writeMHA(out_file, image_list, header_info, spacing_mm, is_compressed):
    header_info['is_compressed'] = is_compressed
    header_info['x_spacing'] = spacing_mm[0]
    header_info['y_spacing'] = spacing_mm[1]
    header_info['z_spacing'] = spacing_mm[2]
    header = HEADER.format(**header_info)

    image_stack = np.hstack(image_list)
    shape = image_stack.shape
    datasize = shape[0]
    # print("im dtype", image_stack.dtype)
    # image_stack = image_stack.reshape((datasize,))
    # image_stack = np.copy(image_stack, 'C')
    with open(out_file, 'wb') as fd:
        # print(header.encode('utf-8'))
        fd.write(header.encode('utf-8'))
        if sys.version_info[0] > 2:
            buf = memoryview(image_stack).tobytes()
        else:
            buf = np.getbuffer(image_stack)
        if is_compressed:
            print("compressing")
            buf = zlib.compress(buf)
        n = fd.write(buf)
        print("wrote", n , len(buf))
# end def

if __name__ == '__main__':
    import argparse
    import os.path
    CMD_DESCRIPTION = """
    Convert a group of TIFF (*.tif) files to an ITK metaimage (*.mha) file
    Expects image files to be named like:

        <inpath>/<prefix>_XXX.tif

    so something like:

        data/CY3_0000.tif
        data/CY3_0001.tif

    etc.

    indexdigits is the number of X's in the XXX
    Z is the slice direction
    """
    parser = argparse.ArgumentParser(
                    prog='tiffs2mha',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description=CMD_DESCRIPTION)
    parser.add_argument('inpath', type=str,
                       help='the path where images are stored')
    parser.add_argument('inprefix', type=str,
                       help='the file name prefix of the tiff image stack')
    parser.add_argument('indexdigits', type=int,
                       help='the number of digits in the filename suffix of the image stack')
    parser.add_argument('outfile', type=str,
                       help='the full path to the output file to write the mha to')
    parser.add_argument('-c', '--compress', action='store_true', default=False,
                       help='whether to zlib compress the data')
    parser.add_argument('-L', '--low', type=int, default=0,
                    help='lower slice index to extract. default is 0.')
    parser.add_argument('-U', '--up', type=int, default=None,
        help='upper slice index to extract. default is the Z dimension of DimSize.')
    parser.add_argument('-s', '--spacing', nargs=3,
        default=(0.0065/40, 0.0065/40, .1),
        metavar=('x', 'y', 'z'),
        help='physical pixel spacing in x, y, z in mm')
    namespace = parser.parse_args()

    in_path = namespace.inpath
    prefix = namespace.inprefix
    out_file = namespace.outfile
    index_digits = namespace.indexdigits
    is_compressed = namespace.compress
    low_index = namespace.low
    upper_index = namespace.up
    spacing_mm = namespace.spacing

    image_list, header_info = stackImages(in_path, prefix, index_digits,
                                low_index=low_index, upper_index=upper_index)
    print("found stack_size:", header_info['stack_size'])
    writeMHA(out_file, image_list, header_info, spacing_mm, is_compressed)
    print('wrote %s' % (out_file))


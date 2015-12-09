import numpy as np
import tifffile as tf
import zlib
import glob

HEADER = """
ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = {0.is_compressed}
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = {0.x_spacing} {0.y_spacing} {0.z_spacing}
DimSize = {image_x_resolution} {image_y_resolution} {stack_size}
ElementType = {datatype}
ElementDataFile = LOCAL
"""

DEFAULT_HEADER_INFO = {
    'is_compressed': False,
    'x_spacing': 0.0065/40,
    'y_spacing': 0.0065/40,
    'z_spacing': 0.100,
    'image_x_resolution': 2560,
    'image_y_resolution': 2160,
}

def stackImages(in_path,
                in_prefix, in_count_format,
                out_file,
                low_index=0,
                upper_index=None,
                header_info=None
                tiff_ext='.tif'
                ):
    """ does not support negative indexing
    in_prefix must be unique for a given image stack
    """
    if header_info is None:
        header_info = DEFAULT_HEADER_INFO.copy()

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

    count_format = '_{:0%d}' % (in_count_format)
    file_str = os.path.join(in_path, in_prefix + count_format + tiff_ext)

    # 3. validate all files in stack exist before writing any file
    for i in range(low_index, upper_index):
        the_file = file_str.format(i)
        if not os.path.isfile(the_file):
            raise ValueError("missing file {}, {} in stack".format(i, the_file))

    # 4. Open and append all tiff files to stack
    image_stack = None
    for i in range(low_index, upper_index):
        the_file = file_str.format(i)
        image = tf.imread(the_file)
        if image_stack is None:
            image_stack = image
        np.vstack((image_stack, image))
    datatype = image.dtype
    if image.dtype.itemsize == 2:
        datatype = 'MET_USHORT'
    elif image.dtype.itemsize == 1:
        datatype = 'MET_UCHAR'
    else:
        raise ValueError("Unsupported bit depth")
    image_x_resolution, image_y_resolution = image.shape
    header = HEADER.format(header_info,
            datatype=datatype,
            stack_size=(upper_index - low_index),
            image_x_resolution=image_x_resolution,
            image_y_resolution=image_y_resolution
            )
    with open(out_file, 'w') as fd:
        fd.write(header)
        image_stack.tofile(fd)
# end def

if __name__ == '__main__':
    import argparse
    import os.path
    CMD_DESCRIPTION = """
    Convert a group of TIFF (*.tif) files to an ITK metaimage (*.mha) file

    """
    parser = argparse.ArgumentParser(
                    prog='tiffs2mha',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description=CMD_DESCRIPTION)
    parser.add_argument('inpath', type=str,
                       help='the path where images are stored')
    parser.add_argument('inprefix', type=str,
                       help='the file name prefix of the tiff image stack')
    parser.add_argument('digitscount', type=str,
                       help='the number of digits in the filename suffix of the image stack')
    parser.add_argument('outfile', type=str,
                       help='the full path to the output file to write the mha to')
    parser.add_argument('-L', '--low', type=int, default=0,
                    help='lower slice index to extract. default is 0.')
    parser.add_argument('-U', '--up', type=int, default=None,
        help='upper slice index to extract. default is the Z dimension of DimSize.')
    namespace = parser.parse_args()

    inpath = namespace.inpath
    prefix = namespace.inprefix
    digits = namespace.digitscount
    root, ext = os.path.splitext(infile)
    low_index = namespace.low
    upper_index = namespace.up
    stackImages(infile, prefix, digits,
            low_index=low_index, upper_index=upper_index)


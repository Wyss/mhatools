import numpy as np
import tifffile as tf
import gzip

LINE_CHECK_LIMIT = 100 # scan up to 100 lines


def splitMHAKeyValue(line):
    line = line.decode('utf-8')
    the_split = line.split(' = ')
    if len(the_split) == 2:
        return  (the_split[0], the_split[1])
    else:
        return None
# end def

def getMHAImageStack(fn, low_index, upper_index):
    """ supports zlib compressed mha files

    reads entire image stack into memory at the moment.
    We could seek to do slicing if we needed to do really large image stacks
    but then we might as well have separate files
    """
    image_stack = None
    dims = None
    is_compressed = False
    with open(fn, 'rb') as fd:
        found_blob = False
        count = 0
        while count < LINE_CHECK_LIMIT:
            line = fd.readline().rstrip()
            res = splitMHAKeyValue(line)
            if res is not None:
                key, val = res
                if key == 'ElementDataFile':
                    found_blob = True
                    break
                elif key == 'DimSize':
                    dims = [int(s) for s in val.split()]
                elif key == 'ElementType':
                    if val == 'MET_UCHAR':
                        datatype = np.dtype('b')
                    elif val == 'MET_USHORT':
                        datatype = np.dtype('<H')
                    else:
                        raise ValueError("unknown METAIMAGE ElementType %s" % (val))
                elif key == 'CompressedData':
                    is_compressed = True if val == 'True' else False
            else:
                raise ValueError("bad MHA file")
            count += 1
        if found_blob:
            """ use fd.read rather than np.fromfile
            to support zlib compressed data
            """
            slice_count = dims[2]
            if upper_index is None:
                upper_index = slice_count

            if abs(upper_index) > slice_count:
                raise ValueError(
                    "Slice upper_index out of range {}".format(upper_index))

            if abs(low_index) > slice_count:
                raise ValueError(
                    "Slice low_index out of range {}".format(low_index))

            # normalize slicing
            if upper_index < 0:
                upper_index += slice_count
            if low_index < 0:
                low_index += slice_count
            if low_index > upper_index:
                raise ValueError(
                    "Normalized low_index,{} is greater than upper_index {}".format(low_index, upper_index))

            data_size = dims[0]*dims[1]*dims[2]
            out_shape = (dims[2], dims[1], dims[0]) # Z axis first
            buf = fd.read(data_size)
            if is_compressed:
                import zlib
                # zlib wbits
                buf = zlib.decompress(buf, 15 + 32)
            image_stack_read = np.frombuffer(buf, dtype=datatype)
            # print("im shape", image_stack_read.shape)

            # copy read data into a new buffer of the appropriate
            # size in case data is truncate as is the case
            # with the Somite0.mha test data from ACME
            if data_size != len(image_stack_read):
                image_stack = np.zeros((data_size,), dtype=datatype)
                image_stack[:len(image_stack_read)] = image_stack_read
            else:
                image_stack = image_stack_read
            image_stack = image_stack.reshape(out_shape)
    return image_stack[low_index:upper_index]

if __name__ == '__main__':
    import argparse
    import os.path
    CMD_DESCRIPTION = """
    Convert an ITK metaimage (*.mha) file to a group of TIFF (*.tif) files
    Assumes the *.mha file uses ElementDataFile=LOCAL

    Slicing uses python standard non-inclusive bounds and supports negative
    indexing
    """
    parser = argparse.ArgumentParser(
                    prog='mha2tiffs',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description=CMD_DESCRIPTION)
    parser.add_argument('infile', type=str,
                       help='the input Metaimage file (*.mha)')
    parser.add_argument('-L', '--low', type=int, default=0,
                    help='lower slice index to extract. default is 0.')
    parser.add_argument('-U', '--up', type=int, default=None,
        help='upper slice index to extract. default is the Z dimension of DimSize.')
    namespace = parser.parse_args()

    infile = namespace.infile
    root, ext = os.path.splitext(infile)
    fout = root + '_s{}.tiff'
    low_index = namespace.low
    upper_index = namespace.up
    istack = getMHAImageStack(infile, low_index, upper_index)

    slice_count = istack.shape[0]
    if low_index < 0:
        low_index = slice_count + low_index
    if upper_index is None:
        upper_index = low_index+slice_count
    for i in range(istack.shape[0]):
        tf.imsave(fout.format(i+low_index), istack[i][:][:])
    print("Converted {} from file: {} between {} and {}".format(
            istack.shape, infile, low_index, upper_index))


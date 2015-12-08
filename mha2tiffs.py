# 9137505
# 9137193
# 311
# 9228800
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

def getMHAImageStack(fn):
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
            data_size = dims[0]*dims[1]*dims[2]
            out_shape = (dims[2], dims[1], dims[0]) # Z axis first
            print("yay!!!!", data_size)
            buf = fd.read(data_size)
            if is_compressed:
                import zlib
                # zlib wbits
                buf = zlib.decompress(buf, 15 + 32)
            image_stack_read = np.frombuffer(buf, dtype=datatype)
            # image_stack_read = np.fromfile(fd, dtype=datatype)
            print("im shape", image_stack_read.shape)
            image_stack = np.zeros((data_size,), dtype=datatype)
            image_stack[:len(image_stack_read)] = image_stack_read
            image_stack = image_stack.reshape(out_shape)
            print("im shape", image_stack.shape)
            print(image_stack[0][:][:].shape)
    return image_stack

if __name__ == '__main__':
    fn = 'test/Somite0.mha'
    istack = getMHAImageStack(fn)
    fout = 'test/Somite0Slice{}.tiff'
    for i in range(2):
        tf.imsave(fout.format(i), istack[i][:][:])
    fn = 'test/Somite0-segment.mha'
    istack = getMHAImageStack(fn)
    fout = 'test/Somite0-segmentSlice{}.tiff'
    for i in range(2):
        tf.imsave(fout.format(i), istack[i][:][:])

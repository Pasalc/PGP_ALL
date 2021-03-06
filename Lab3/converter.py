import sys
import struct
from PIL import Image
import ctypes

pal = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255)
]


def to_img_alfa(src, dst=None):
    fin = open(src, 'rb')
    (w, h) = struct.unpack('ii', fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new('RGB', (w, h))
    pix = img.load()
    offset = 0
    sp = len(pal)
    for j in range(h):
        for i in range(w):
            (_, _, _, a) = struct.unpack_from('BBBB', buff, offset)
            pix[i, j] = pal[a]
            offset += 4
    if dst:
        img.save(dst)
    else:
        img.show()


def to_img(src, dst=None):
    fin = open(src, 'rb')
    (w, h) = struct.unpack('ii', fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new('RGB', (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            (r, g, b, _) = struct.unpack_from('BBBB', buff, offset)
            pix[i, j] = (r, g, b)
            offset += 4
    if dst:
        img.save(dst)
    else:
        img.show()


def from_img(src, dst):
    img = Image.open(src)
    (w, h) = img.size[0:2]
    print(w,h)
    pix = img.load()
    buff = ctypes.create_string_buffer(4 * w * h)
    offset = 0
    for j in range(h):
        for i in range(w):
            r = pix[i, j][0]
            g = pix[i, j][1]
            b = pix[i, j][2]
            struct.pack_into('BBBB', buff, offset, r, g, b, 0)
            offset += 4;
    out = open(dst, 'wb')
    out.write(struct.pack('ii', w, h))
    out.write(buff.raw)
    out.close()


def main():
    argv = []
    alfa = show = False
    for arg in sys.argv[1:]:
        if arg == '-a':
            alfa = True
        elif arg == '-s':
            show = True
        else:
            argv.append(arg)
    if show:
        if len(argv) == 1 and argv[0].endswith('.data'):
            if alfa:
                to_img_alfa(argv[0])
            else:
                to_img(argv[0])
        else:
            print('Error argv')
    else:
        if len(argv) != 2:
            print('Error count argv')
            sys.exit(1)
        f1 = argv[0].endswith('.data')
        f2 = argv[1].endswith('.data')
        if not (f1 or f2):
            print('Error type argv or count')
            sys.exit(1)
        if f1:
            if alfa:
                to_img_alfa(argv[0], argv[1])
            else:
                to_img(argv[0], argv[1])
        else:
            if not alfa:
                from_img(argv[0], argv[1])
            else:
                print('Error alfa')


if __name__ == '__main__':
    main()
import argparse
import igl


def main():
    parser = argparse.ArgumentParser(description='Center and normalize a mesh.')
    parser.add_argument('input', help='Input mesh file (e.g. .obj)')
    parser.add_argument('output', help='Output mesh file (e.g. .off, .obj, .ply)')
    args = parser.parse_args()

    v, f = igl.read_triangle_mesh(args.input)

    bb_max = v.max(axis=0, keepdims=True)
    bb_min = v.min(axis=0, keepdims=True)

    centers = (bb_max + bb_min) / 2.0
    v = v - centers
    v = v / (bb_max - bb_min).max()

    print(centers)
    print(bb_max - bb_min)

    igl.write_triangle_mesh(args.output, v, f)
    print('Finished: {}'.format(args.input))


if __name__ == '__main__':
    main()
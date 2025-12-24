import sqlite3
import struct
import sys
from collections import Counter


def decode_vec(v):
    """Decode a 3D vector stored as BLOB (3x float32/float64) or TEXT 'x y z'."""
    if v is None:
        return None
    if isinstance(v, memoryview):
        v = v.tobytes()
    if isinstance(v, str):
        parts = v.strip().split()
        if len(parts) == 3:
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                return None
        return None
    if not isinstance(v, (bytes, bytearray)):
        return None
    if len(v) == 24:
        return struct.unpack('<ddd', v)
    if len(v) == 12:
        return struct.unpack('<fff', v)
    return None


def main():
    if len(sys.argv) < 2:
        print('Usage: python check_colmap_priors.py <path_to_database.db>')
        sys.exit(1)

    db = sys.argv[1]
    con = sqlite3.connect(db)
    cur = con.cursor()

    cur.execute('pragma table_info(images)')
    cols = [c[1] for c in cur.fetchall()]
    print('images columns:', cols)
    interesting = [c for c in cols if any(k in c.lower() for k in ['prior', 'tvec', 'gps', 'lat', 'lon', 'alt', 'pose', 'origin'])]
    print('interesting columns:', interesting)

    if 'tvec_prior' not in cols:
        print('No tvec_prior in images table (model_aligner may have nothing to align).')
        sys.exit(0)

    cur.execute('select count(*), sum(tvec_prior is not null), sum(length(tvec_prior) > 0) from images')
    print('count / tvec_prior notnull / tvec_prior len>0:', cur.fetchone())

    cur.execute('select name, tvec_prior from images where tvec_prior is not null and length(tvec_prior) > 0')
    rows = cur.fetchall()

    decoded = []
    zeros = 0
    for name, blob in rows:
        vec = decode_vec(blob)
        if vec is None:
            continue
        decoded.append(vec)
        if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
            zeros += 1

    print('decoded tvec_prior:', len(decoded), 'entries')
    print('exact (0,0,0) entries:', zeros)

    if not decoded:
        print('No decodable tvec_prior values found.')
        sys.exit(0)

    xs = [v[0] for v in decoded]
    ys = [v[1] for v in decoded]
    zs = [v[2] for v in decoded]
    print('range x:', (min(xs), max(xs)))
    print('range y:', (min(ys), max(ys)))
    print('range z:', (min(zs), max(zs)))

    # show 10 most common rounded positions to detect degeneracy
    rounded = Counter((round(v[0], 6), round(v[1], 6), round(v[2], 3)) for v in decoded)
    print('top 10 most common (rounded) positions:')
    for pos, cnt in rounded.most_common(10):
        print(' ', pos, 'x', cnt)


if __name__ == '__main__':
    main()

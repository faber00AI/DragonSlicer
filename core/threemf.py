"""Write a standard 3MF (core spec only): one object per part, plate coordinates in millimetres, identity build transforms.
Standard library + numpy, no lxml. Streams the XML so multi-million-face parts do not need a second copy in memory."""
import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
RELS = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
BLOCK = 500_000


def _rows(fh, a, fmt):
    for i in range(0, len(a), BLOCK):
        buf = io.StringIO()
        np.savetxt(buf, a[i:i + BLOCK], fmt=fmt, newline="\n")
        fh.write(buf.getvalue().encode())


def write_3mf(out, parts) -> Path:
    """`parts`: [(name, trimesh.Trimesh)] already placed in plate coordinates."""
    out = Path(out)
    if out.suffix.lower() != ".3mf":
        out = out.with_suffix(".3mf")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        with z.open("3D/3dmodel.model", "w", force_zip64=True) as fh:
            fh.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="{CORE}">\n<resources>\n'.encode())
            for i, (name, mesh) in enumerate(parts, 1):
                fh.write(f'<object id="{i}" name={quoteattr(name)} type="model"><mesh><vertices>\n'.encode())
                _rows(fh, np.asarray(mesh.vertices, float), '<vertex x="%.6f" y="%.6f" z="%.6f"/>')
                fh.write(b"</vertices><triangles>\n")
                _rows(fh, np.asarray(mesh.faces, np.int64), '<triangle v1="%d" v2="%d" v3="%d"/>')
                fh.write(b"</triangles></mesh></object>\n")
            fh.write(b"</resources>\n<build>\n")
            for i in range(1, len(parts) + 1):
                fh.write(f'<item objectid="{i}"/>\n'.encode())
            fh.write(b"</build>\n</model>\n")
    return out

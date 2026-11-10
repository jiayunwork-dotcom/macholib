"""
Pure-python readers for the dyld fixup information and the export trie
embedded in 64-bit Mach-O headers.

The readers only inspect data that is reachable through the load commands
that were already parsed by :class:`macholib.MachO.MachOHeader`.  All file
offsets in those load commands are relative to the slice the header belongs
to, therefore every read goes through a :class:`macholib.util.fileview`
window for that slice, never through a whole-file offset.

Only the generic 64-bit chained pointer formats
(``DYLD_CHAINED_PTR_64`` and ``DYLD_CHAINED_PTR_64_OFFSET``) are understood.
Authenticated arm64e pointers and 32-bit chained pointers are out of scope
and ignored.
"""

import struct

from .mach_o import (
    BIND_IMMEDIATE_MASK,
    BIND_OPCODE_ADD_ADDR_ULEB,
    BIND_OPCODE_DO_BIND,
    BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED,
    BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB,
    BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB,
    BIND_OPCODE_DONE,
    BIND_OPCODE_MASK,
    BIND_OPCODE_SET_ADDEND_SLEB,
    BIND_OPCODE_SET_DYLIB_ORDINAL_IMM,
    BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB,
    BIND_OPCODE_SET_DYLIB_SPECIAL_IMM,
    BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB,
    BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM,
    BIND_OPCODE_SET_TYPE_IMM,
    EXPORT_SYMBOL_FLAGS_REEXPORT,
    EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER,
    LC_DYLD_CHAINED_FIXUPS,
    LC_DYLD_EXPORTS_TRIE,
    LC_DYLD_INFO,
    LC_DYLD_INFO_ONLY,
    LC_LAZY_LOAD_DYLIB,
    LC_LOAD_DYLIB,
    LC_LOAD_UPWARD_DYLIB,
    LC_LOAD_WEAK_DYLIB,
    LC_REEXPORT_DYLIB,
    LC_SEGMENT_64,
    MH_CIGAM_64,
    MH_MAGIC_64,
    REBASE_IMMEDIATE_MASK,
    REBASE_OPCODE_ADD_ADDR_IMM_SCALED,
    REBASE_OPCODE_ADD_ADDR_ULEB,
    REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB,
    REBASE_OPCODE_DO_REBASE_IMM_TIMES,
    REBASE_OPCODE_DO_REBASE_ULEB_TIMES,
    REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB,
    REBASE_OPCODE_DONE,
    REBASE_OPCODE_MASK,
    REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB,
    REBASE_OPCODE_SET_TYPE_IMM,
    REBASE_TYPE_POINTER,
    REBASE_TYPE_TEXT_ABSOLUTE32,
    REBASE_TYPE_TEXT_PCREL32,
)
from .ptypes import sizeof
from .util import fileview

# Values from <mach-o/fixup-chains.h>, which are not part of mach_o.py.
DYLD_CHAINED_PTR_64 = 2
DYLD_CHAINED_PTR_64_OFFSET = 6
DYLD_CHAINED_IMPORT = 1
DYLD_CHAINED_IMPORT_ADDEND = 2
DYLD_CHAINED_IMPORT_ADDEND64 = 3
_DYLD_CHAINED_PTR_START_NONE = 0xFFFF
_DYLD_CHAINED_PTR_START_MULTI = 0x8000

# Load commands that consume a (positive) dylib ordinal.  LC_ID_DYLIB is
# intentionally excluded: it identifies the image itself, not a dependency.
_DEPENDENCY_CMDS = frozenset(
    [
        LC_LOAD_DYLIB,
        LC_LOAD_WEAK_DYLIB,
        LC_REEXPORT_DYLIB,
        LC_LAZY_LOAD_DYLIB,
        LC_LOAD_UPWARD_DYLIB,
    ]
)


def is_64bit_header(header):
    """Return True when *header* is a 64-bit Mach-O header."""
    return header.MH_MAGIC in (MH_MAGIC_64, MH_CIGAM_64)


def read_fixups(header):
    """Return the fixup entries contained in a 64-bit *header*."""
    if not is_64bit_header(header):
        return []
    view = _HeaderView.open(header)
    try:
        entries = []
        entries.extend(_read_classic_fixups(view, header))
        entries.extend(_read_chained_fixups(view, header))
        return entries
    finally:
        view.close()


def read_exports(header):
    """Return the exported symbols described by the trie of *header*."""
    if not is_64bit_header(header):
        return []
    view = _HeaderView.open(header)
    try:
        trie = _locate_export_trie(view, header)
        if trie is None:
            return []
        dylibs = view.dylibs
        return _walk_export_trie(trie, dylibs)
    finally:
        view.close()


class _HeaderView(object):
    """
    Read-only window on the slice backing a MachOHeader together with the
    segment, section and dependency information derived from its load
    commands.
    """

    def __init__(self, header, fileobj, raw_file):
        self.header = header
        self.fileobj = fileobj
        self.raw_file = raw_file
        self.endian = header.endian
        self.segments = []
        self.sections = []
        self.dylibs = {}
        self._collect_load_command_data(header)
        # The image base is the preferred load address of the Mach-O
        # header, which is the vmaddr of the file-backed segment starting
        # at file offset 0 (normally __TEXT).  Executables put __PAGEZERO
        # at vmaddr 0 while dylibs legitimately map __TEXT at vmaddr 0, so
        # simply taking the smallest vmaddr is not correct.
        self.image_base = 0
        for segment in self.segments:
            if segment["fileoff"] == 0 and segment["filesize"] != 0:
                self.image_base = segment["vmaddr"]
                break

    @classmethod
    def open(cls, header):
        raw_file = open(header.parent.filename, "rb")
        window = fileview(raw_file, header.offset, header.size)
        return cls(header, window, raw_file)

    def close(self):
        self.raw_file.close()

    def _collect_load_command_data(self, header):
        ordinal = 0
        for load, command, data in header.commands:
            cmd = load.cmd
            if cmd == LC_SEGMENT_64:
                segname = _name_from_field(command.segname)
                segment = {
                    "name": segname,
                    "vmaddr": int(command.vmaddr),
                    "vmsize": int(command.vmsize),
                    "fileoff": int(command.fileoff),
                    "filesize": int(command.filesize),
                }
                self.segments.append(segment)
                for section in data:
                    self.sections.append(
                        {
                            "segment": _name_from_field(section.segname),
                            "name": _name_from_field(section.sectname),
                            "addr": int(section.addr),
                            "size": int(section.size),
                        }
                    )
            elif cmd in _DEPENDENCY_CMDS:
                ordinal += 1
                self.dylibs[ordinal] = _lc_str_string(load, command, data)

    def locate(self, address):
        """Map a virtual address to a (segment name, section name) pair."""
        segment_name = None
        for segment in self.segments:
            start = segment["vmaddr"]
            if start <= address < start + segment["vmsize"]:
                segment_name = segment["name"]
                break
        section_name = None
        for section in self.sections:
            start = section["addr"]
            if start <= address < start + section["size"]:
                section_name = section["name"]
                break
        return segment_name, section_name

    def vmaddr_to_fileoff(self, address):
        """
        Translate a virtual address to a slice-relative file offset, or
        return None when the address is not backed by file content.
        """
        for segment in self.segments:
            start = segment["vmaddr"]
            if start <= address < start + segment["filesize"]:
                return segment["fileoff"] + (address - start)
        return None

    def read_uint(self, fileoff, size, endian=None, signed=False):
        if endian is None:
            endian = self.endian
        if size == 8:
            fmt = "q" if signed else "Q"
        elif size == 4:
            fmt = "i" if signed else "I"
        elif size == 2:
            fmt = "h" if signed else "H"
        else:
            fmt = "b" if signed else "B"
        self.fileobj.seek(fileoff)
        raw = self.fileobj.read(size)
        if len(raw) != size:
            return None
        return struct.unpack(endian + fmt, raw)[0]


def _name_from_field(value):
    # p_str16 values are NUL padded bytes in both Python 2 and 3.
    return value.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _lc_str_string(load, command, data):
    offset = command.name - sizeof(load.__class__) - sizeof(command.__class__)
    end = data.find(b"\x00", offset)
    if end < 0:
        end = len(data)
    return data[offset:end].decode("utf-8", "replace")


def _make_fixup(
    segment,
    section,
    address,
    kind,
    target=None,
    dylib=None,
    dylib_ordinal=None,
    symbol=None,
    addend=None,
):
    return {
        "segment": segment,
        "section": section,
        "address": address,
        "kind": kind,
        "target": target,
        "dylib": dylib,
        "dylib_ordinal": dylib_ordinal,
        "symbol": symbol,
        "addend": addend,
    }


def _make_export(name, flags, address, reexport_dylib, reexport_symbol):
    return {
        "name": name,
        "flags": flags,
        "address": address,
        "reexport_dylib": reexport_dylib,
        "reexport_symbol": reexport_symbol,
    }


def _find_command(header, *cmds):
    for load, command, data in header.commands:
        if load.cmd in cmds:
            return load, command, data
    return None


# ---------------------------------------------------------------------------
# Classic LC_DYLD_INFO[_ONLY] opcode streams
# ---------------------------------------------------------------------------


def _read_classic_fixups(view, header):
    info = _find_command(header, LC_DYLD_INFO, LC_DYLD_INFO_ONLY)
    if info is None:
        return []
    _load, command, _data = info
    entries = []
    entries.extend(
        _parse_rebase_stream(
            view,
            int(command.rebase_off),
            int(command.rebase_size),
        )
    )
    entries.extend(
        _parse_bind_stream(
            view,
            int(command.bind_off),
            int(command.bind_size),
            "bind",
        )
    )
    entries.extend(
        _parse_bind_stream(
            view,
            int(command.weak_bind_off),
            int(command.weak_bind_size),
            "weak_bind",
        )
    )
    entries.extend(
        _parse_bind_stream(
            view,
            int(command.lazy_bind_off),
            int(command.lazy_bind_size),
            "lazy_bind",
        )
    )
    return entries


def _read_slice(view, offset, size):
    if size == 0:
        return b""
    view.fileobj.seek(offset)
    return bytes(view.fileobj.read(size))


def _segment_vmaddr(view, segment_index):
    if 0 <= segment_index < len(view.segments):
        return view.segments[segment_index]["vmaddr"]
    return None


def _parse_rebase_stream(view, offset, size):
    if size == 0:
        return []
    blob = _read_slice(view, offset, size)
    reader = _ByteReader(blob)
    entries = []
    rebase_type = REBASE_TYPE_POINTER
    segment_index = 0
    segment_offset = 0

    def emit(count, extra_step):
        # Every rebase action implicitly advances the location by one
        # pointer width; extra_step is the additional delta of the opcode.
        base = _segment_vmaddr(view, segment_index)
        if base is None:
            return
        current = segment_offset
        for _ in range(count):
            address = base + current
            target = _read_rebase_target(view, address, rebase_type)
            segment_name, section_name = view.locate(address)
            entries.append(
                _make_fixup(
                    segment_name,
                    section_name,
                    address,
                    "rebase",
                    target=target,
                )
            )
            current += 8 + extra_step

    while True:
        byte = reader.u8()
        if byte is None:
            break
        opcode = byte & REBASE_OPCODE_MASK
        immediate = byte & REBASE_IMMEDIATE_MASK
        if opcode == REBASE_OPCODE_DONE:
            break
        elif opcode == REBASE_OPCODE_SET_TYPE_IMM:
            rebase_type = immediate
        elif opcode == REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB:
            segment_index = immediate
            segment_offset = reader.uleb()
        elif opcode == REBASE_OPCODE_ADD_ADDR_ULEB:
            segment_offset += reader.uleb()
        elif opcode == REBASE_OPCODE_ADD_ADDR_IMM_SCALED:
            segment_offset += immediate * 8
        elif opcode == REBASE_OPCODE_DO_REBASE_IMM_TIMES:
            emit(immediate, 0)
            segment_offset += immediate * 8
        elif opcode == REBASE_OPCODE_DO_REBASE_ULEB_TIMES:
            count = reader.uleb()
            emit(count, 0)
            segment_offset += count * 8
        elif opcode == REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB:
            extra = reader.uleb()
            emit(1, extra)
            segment_offset += 8 + extra
        elif opcode == REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB:
            count = reader.uleb()
            skip = reader.uleb()
            emit(count, skip)
            segment_offset += count * (8 + skip)
        else:
            # Unknown opcode (e.g. threaded binds): the stream cannot be
            # interpreted by this reader, stop where the opcode starts.
            break
    return entries


def _read_rebase_target(view, address, rebase_type):
    fileoff = view.vmaddr_to_fileoff(address)
    if fileoff is None:
        return None
    if rebase_type == REBASE_TYPE_TEXT_ABSOLUTE32:
        return view.read_uint(fileoff, 4)
    if rebase_type == REBASE_TYPE_TEXT_PCREL32:
        relative = view.read_uint(fileoff, 4, signed=True)
        if relative is None:
            return None
        return address + 4 + relative
    return view.read_uint(fileoff, 8)


def _parse_bind_stream(view, offset, size, kind):
    if size == 0:
        return []
    blob = _read_slice(view, offset, size)
    reader = _ByteReader(blob)
    entries = []
    ordinal = 0
    symbol = None
    addend = 0
    segment_index = 0
    segment_offset = 0
    # The lazy bind blob is a concatenation of one DONE terminated record
    # per lazy pointer, so DONE only ends a single record there.  The
    # regular and weak bind streams are a single stream ending in DONE.
    done_ends_stream = kind != "lazy_bind"

    def emit(count, extra_step):
        # Every bind action implicitly advances the location by one
        # pointer width; extra_step is the additional delta of the opcode.
        base = _segment_vmaddr(view, segment_index)
        if base is None:
            return
        current = segment_offset
        for _ in range(count):
            address = base + current
            segment_name, section_name = view.locate(address)
            if ordinal > 0:
                dylib = view.dylibs.get(ordinal)
            else:
                dylib = None
            entries.append(
                _make_fixup(
                    segment_name,
                    section_name,
                    address,
                    kind,
                    dylib=dylib,
                    dylib_ordinal=ordinal,
                    symbol=symbol,
                    addend=addend,
                )
            )
            current += 8 + extra_step

    while True:
        byte = reader.u8()
        if byte is None:
            break
        opcode = byte & BIND_OPCODE_MASK
        immediate = byte & BIND_IMMEDIATE_MASK
        if opcode == BIND_OPCODE_DONE:
            if done_ends_stream:
                break
            continue
        elif opcode == BIND_OPCODE_SET_DYLIB_ORDINAL_IMM:
            ordinal = immediate
        elif opcode == BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB:
            ordinal = reader.uleb()
        elif opcode == BIND_OPCODE_SET_DYLIB_SPECIAL_IMM:
            if immediate == 0:
                ordinal = 0
            else:
                ordinal = immediate - 16
        elif opcode == BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM:
            symbol = reader.cstring().decode("utf-8", "replace")
        elif opcode == BIND_OPCODE_SET_TYPE_IMM:
            # The bind type is not needed beyond consuming the opcode.
            pass
        elif opcode == BIND_OPCODE_SET_ADDEND_SLEB:
            addend = reader.sleb()
        elif opcode == BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB:
            segment_index = immediate
            segment_offset = reader.uleb()
        elif opcode == BIND_OPCODE_ADD_ADDR_ULEB:
            segment_offset += reader.uleb()
        elif opcode == BIND_OPCODE_DO_BIND:
            emit(1, 0)
            segment_offset += 8
        elif opcode == BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB:
            extra = reader.uleb()
            emit(1, extra)
            segment_offset += 8 + extra
        elif opcode == BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED:
            extra = immediate * 8
            emit(1, extra)
            segment_offset += 8 + extra
        elif opcode == BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB:
            count = reader.uleb()
            skip = reader.uleb()
            emit(count, skip)
            segment_offset += count * (8 + skip)
        else:
            # BIND_OPCODE_THREADED and any future opcode are out of scope.
            break
    return entries


# ---------------------------------------------------------------------------
# LC_DYLD_CHAINED_FIXUPS
# ---------------------------------------------------------------------------


def _read_chained_fixups(view, header):
    command_info = _find_command(header, LC_DYLD_CHAINED_FIXUPS)
    if command_info is None:
        return []
    _load, command, _data = command_info
    dataoff = int(command.dataoff)
    datasize = int(command.datasize)
    blob = _read_slice(view, dataoff, datasize)
    if len(blob) < 28:
        return []
    (
        _fixups_version,
        starts_offset,
        imports_offset,
        symbols_offset,
        imports_count,
        imports_format,
        symbols_format,
    ) = struct.unpack_from("<7I", blob, 0)

    imports = _parse_chained_imports(
        blob,
        imports_offset,
        imports_count,
        imports_format,
        symbols_offset,
        symbols_format,
    )

    if len(blob) < starts_offset + 4:
        return []
    seg_count = struct.unpack_from("<I", blob, starts_offset)[0]
    seg_info_offsets = struct.unpack_from("<%dI" % seg_count, blob, starts_offset + 4)

    entries = []
    for _seg_index, seg_info_offset in enumerate(seg_info_offsets):
        if seg_info_offset == 0:
            continue
        entries.extend(
            _walk_chained_segment(
                view,
                blob,
                starts_offset + seg_info_offset,
                imports,
            )
        )
    return entries


def _decode_chained_ordinal(raw, bits):
    if raw == 0:
        return 0
    sign_base = 1 << bits
    # The top 15 encodings represent the negative special ordinals, see
    # the lib_ordinal comments in <mach-o/fixup-chains.h>.
    if raw >= sign_base - 15:
        return raw - sign_base
    return raw


def _parse_chained_imports(
    blob, imports_offset, count, imports_format, symbols_offset, symbols_format
):
    imports = []
    if symbols_format == 0:
        symbols_pool = blob[symbols_offset:]
    else:
        # Compressed symbol pools are out of scope; names resolve to None.
        symbols_pool = None
    pos = imports_offset
    for _ in range(count):
        if imports_format == DYLD_CHAINED_IMPORT_ADDEND64:
            if pos + 16 > len(blob):
                break
            word0, addend_raw = struct.unpack_from("<QQ", blob, pos)
            pos += 16
            raw_ordinal = word0 & 0xFFFF
            name_offset = (word0 >> 32) & 0xFFFFFFFF
            ordinal = _decode_chained_ordinal(raw_ordinal, 16)
            addend = struct.unpack("<q", struct.pack("<Q", addend_raw))[0]
        else:
            if pos + 4 > len(blob):
                break
            (word,) = struct.unpack_from("<I", blob, pos)
            pos += 4
            raw_ordinal = word & 0xFF
            name_offset = (word >> 9) & 0x7FFFFF
            ordinal = _decode_chained_ordinal(raw_ordinal, 8)
            if imports_format == DYLD_CHAINED_IMPORT_ADDEND:
                if pos + 4 > len(blob):
                    break
                (addend,) = struct.unpack_from("<i", blob, pos)
                pos += 4
            else:
                addend = 0
        name = None
        if symbols_pool is not None:
            name = _cstring_from(symbols_pool, name_offset)
        imports.append({"ordinal": ordinal, "addend": addend, "symbol": name})
    return imports


def _cstring_from(blob, offset):
    end = blob.find(b"\x00", offset)
    if end < 0:
        end = len(blob)
    return bytes(blob[offset:end]).decode("utf-8", "replace")


def _walk_chained_segment(view, blob, seg_info_start, imports):
    fixed_header = struct.unpack_from("<IHHQIH", blob, seg_info_start)
    (
        _size,
        page_size,
        pointer_format,
        segment_offset,
        _max_valid_pointer,
        page_count,
    ) = fixed_header
    if pointer_format not in (
        DYLD_CHAINED_PTR_64,
        DYLD_CHAINED_PTR_64_OFFSET,
    ):
        # Authenticated arm64e and 32-bit chain formats are not supported.
        return []

    page_starts_start = seg_info_start + 22
    entries = []
    for page_index in range(page_count):
        position = page_starts_start + page_index * 2
        if position + 2 > len(blob):
            break
        (page_start,) = struct.unpack_from("<H", blob, position)
        if page_start == _DYLD_CHAINED_PTR_START_NONE:
            continue
        if page_start & _DYLD_CHAINED_PTR_START_MULTI:
            # Multiple starts per page only occur in unsupported formats.
            continue
        chain_offset = segment_offset + page_index * page_size + page_start
        entries.extend(
            _walk_pointer_chain(
                view,
                chain_offset,
                pointer_format,
                imports,
            )
        )
    return entries


def _walk_pointer_chain(view, chain_offset, pointer_format, imports):
    entries = []
    while True:
        address = view.image_base + chain_offset
        fileoff = view.vmaddr_to_fileoff(address)
        if fileoff is None:
            break
        pointer = view.read_uint(fileoff, 8, endian="<")
        if pointer is None:
            break
        segment_name, section_name = view.locate(address)
        if pointer & (1 << 63):
            entry = _decode_chained_bind(
                pointer, imports, segment_name, section_name, address, view
            )
            if entry is not None:
                entries.append(entry)
        else:
            target = pointer & 0xFFFFFFFFF
            target |= ((pointer >> 36) & 0xFF) << 56
            if pointer_format == DYLD_CHAINED_PTR_64_OFFSET:
                target += view.image_base
            entries.append(
                _make_fixup(
                    segment_name,
                    section_name,
                    address,
                    "chained_rebase",
                    target=target,
                )
            )
        next_stride = (pointer >> 51) & 0xFFF
        if next_stride == 0:
            break
        # The "next" field is a 4-byte stride, see the fixup-chains.h notes.
        chain_offset += next_stride * 4
    return entries


def _decode_chained_bind(pointer, imports, segment_name, section_name, address, view):
    import_index = pointer & 0xFFFFFF
    pointer_addend = (pointer >> 24) & 0xFF
    if import_index >= len(imports):
        return None
    imported = imports[import_index]
    ordinal = imported["ordinal"]
    if ordinal > 0:
        dylib = view.dylibs.get(ordinal)
    else:
        dylib = None
    return _make_fixup(
        segment_name,
        section_name,
        address,
        "chained_bind",
        dylib=dylib,
        dylib_ordinal=ordinal,
        symbol=imported["symbol"],
        addend=imported["addend"] + pointer_addend,
    )


# ---------------------------------------------------------------------------
# Export trie
# ---------------------------------------------------------------------------


def _locate_export_trie(view, header):
    info = _find_command(header, LC_DYLD_INFO, LC_DYLD_INFO_ONLY)
    if info is not None:
        _load, command, _data = info
        if int(command.export_size) != 0:
            return _read_slice(
                view,
                int(command.export_off),
                int(command.export_size),
            )
    command_info = _find_command(header, LC_DYLD_EXPORTS_TRIE)
    if command_info is not None:
        _load, command, _data = command_info
        if int(command.datasize) != 0:
            return _read_slice(
                view,
                int(command.dataoff),
                int(command.datasize),
            )
    return None


def _walk_export_trie(trie, dylibs):
    results = []
    stack = [(0, "")]
    while stack:
        node_offset, prefix = stack.pop()
        reader = _ByteReader(trie, node_offset)
        terminal_size = reader.uleb()
        terminal_info = None
        if terminal_size != 0:
            terminal_start = reader.pos
            flags = reader.uleb()
            terminal_info = _read_terminal_info(reader, flags, prefix, dylibs)
            # Defensive: terminal_size covers the whole terminal node body.
            reader.pos = terminal_start + terminal_size
        child_count = reader.u8()
        children = []
        for _ in range(child_count):
            edge = reader.cstring().decode("utf-8", "replace")
            child_offset = reader.uleb()
            children.append((child_offset, prefix + edge))
        # Reversed pushes keep the on-disk edge order when popping.
        for child in reversed(children):
            stack.append(child)
        if terminal_info is not None:
            results.append(terminal_info)
    return results


def _read_terminal_info(reader, flags, symbol_name, dylibs):
    if flags & EXPORT_SYMBOL_FLAGS_REEXPORT:
        ordinal = reader.uleb()
        imported_name = reader.cstring().decode("utf-8", "replace")
        if ordinal > 0:
            reexport_dylib = dylibs.get(ordinal)
        else:
            reexport_dylib = None
        reexport_symbol = imported_name if imported_name else symbol_name
        return _make_export(symbol_name, flags, None, reexport_dylib, reexport_symbol)
    if flags & EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER:
        address = reader.uleb()
        # The resolver offset is the second uleb and is not exposed here.
        reader.uleb()
    else:
        # Regular exports store an image relative offset; absolute exports
        # (EXPORT_SYMBOL_FLAGS_KIND_ABSOLUTE) already carry the absolute
        # value, so in both cases the raw uleb is the requested address.
        address = reader.uleb()
    return _make_export(symbol_name, flags, address, None, None)


# ---------------------------------------------------------------------------
# ULEB128 / SLEB128 helper
# ---------------------------------------------------------------------------


class _ByteReader(object):
    def __init__(self, data, position=0):
        self.data = bytearray(data)
        self.pos = position

    def u8(self):
        if self.pos >= len(self.data):
            return None
        value = self.data[self.pos]
        self.pos += 1
        return value

    def uleb(self):
        result = 0
        shift = 0
        while True:
            byte = self.u8()
            if byte is None:
                return result
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                return result
            shift += 7

    def sleb(self):
        result = 0
        shift = 0
        while True:
            byte = self.u8()
            if byte is None:
                return result
            result |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                if shift < 64 and (byte & 0x40):
                    result |= -(1 << shift)
                return result

    def cstring(self):
        start = self.pos
        end = self.data.find(b"\x00", start)
        if end < 0:
            end = len(self.data)
        value = bytes(self.data[start:end])
        self.pos = end + 1
        return value

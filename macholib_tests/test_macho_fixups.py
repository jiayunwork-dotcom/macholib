"""
Tests for MachOHeader.getFixups() and MachOHeader.getExports().

The Mach-O files used here are synthesized from scratch so that every
opcode, chained pointer and export trie edge can be exercised, including
the variants that are hard to find in real binaries.
"""

import hashlib
import os
import struct
import sys
import tempfile

from macholib import MachO, mach_o
from macholib.macho_fixups import DYLD_CHAINED_PTR_64, DYLD_CHAINED_PTR_64_OFFSET

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest


# ---------------------------------------------------------------------------
# Basic encoding helpers
# ---------------------------------------------------------------------------


def uleb(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def sleb(value):
    out = bytearray()
    more = True
    while more:
        byte = value & 0x7F
        value >>= 7
        sign_bit = byte & 0x40
        if (value == 0 and not sign_bit) or (value == -1 and sign_bit):
            more = False
            out.append(byte)
        else:
            out.append(byte | 0x80)
    return bytes(out)


def cstring(value):
    return value.encode("utf-8") + b"\x00"


# ---------------------------------------------------------------------------
# Mach-O fixture builder
# ---------------------------------------------------------------------------

BASE = 0x100000000
DATA_FILE_OFF = 0x2000
DATA_VM_OFF = 0x10000
DATA_FILE_SIZE = 0x2000
BLOB_OFF = 0x4000
FILE_SIZE = 0x6000

GOT_OFF = 0x10000
LAZY_OFF = 0x10080
CONST_OFF = 0x10100
# 0x10200 .. end of segment belongs to no section.

LC_DYLD_INFO_ONLY = 0x80000022
LC_CHAINED = mach_o.LC_DYLD_CHAINED_FIXUPS
LC_EXPORTS_TRIE = mach_o.LC_DYLD_EXPORTS_TRIE
# EXPORT_SYMBOL_FLAGS_KIND_ABSOLUTE is not exported by mach_o.mach_o.
EXPORT_KIND_ABSOLUTE = 0x02


def section_64(sectname, segname, addr, size, offset):
    return struct.pack(
        "<16s16sQQIIIIIIII",
        sectname.encode("utf-8").ljust(16, b"\x00"),
        segname.encode("utf-8").ljust(16, b"\x00"),
        addr,
        size,
        offset,
        3,  # align = 2**3
        0,
        0,
        0,
        0,
        0,
        0,
    )


def segment_64(name, vmaddr, vmsize, fileoff, filesize, sections):
    cmdsize = 72 + sum(len(s) for s in sections)
    return struct.pack(
        "<II16sQQQQiiII",
        mach_o.LC_SEGMENT_64,
        cmdsize,
        name.encode("utf-8").ljust(16, b"\x00"),
        vmaddr,
        vmsize,
        fileoff,
        filesize,
        7,  # maxprot rwx
        5,  # initprot rx
        len(sections),
        0,
    ) + b"".join(sections)


def load_dylib(install_name):
    name_bytes = install_name.encode("utf-8") + b"\x00"
    cmdsize = 24 + len(name_bytes)
    padding = (-cmdsize) % 8
    cmdsize += padding
    return (
        struct.pack(
            "<IIIIII",
            mach_o.LC_LOAD_DYLIB,
            cmdsize,
            24,
            2,
            0x10000,
            0,
        )
        + name_bytes
        + b"\x00" * padding
    )


def dyld_info_command(streams, export_off=0, export_size=0):
    return struct.pack(
        "<12I",
        LC_DYLD_INFO_ONLY,
        48,
        streams.get("rebase_off", 0),
        streams.get("rebase_size", 0),
        streams.get("bind_off", 0),
        streams.get("bind_size", 0),
        streams.get("weak_off", 0),
        streams.get("weak_size", 0),
        streams.get("lazy_off", 0),
        streams.get("lazy_size", 0),
        export_off,
        export_size,
    )


def linkedit_command(cmd, dataoff, datasize):
    return struct.pack("<4I", cmd, 16, dataoff, datasize)


class MachOBuilder(object):
    def __init__(self, cputype=0x0100000C, magic=mach_o.MH_MAGIC_64):
        # Everything in the fixture is little endian; the on-disk magic
        # bytes therefore spell MH_MAGIC_64 in LE, which macholib reads as
        # MH_CIGAM_64 and parses with endianness "<".
        self.cputype = cputype
        self.magic = magic
        self.commands = []
        self.data = bytearray(FILE_SIZE)
        self._add_default_segments()
        self._add_default_dylibs()

    def _add_default_segments(self):
        text_section = section_64("__text", "__TEXT", BASE + 0x400, 0x100, 0x400)
        self.text_segment = segment_64(
            "__TEXT",
            BASE,
            0x10000,
            0,
            DATA_FILE_OFF,
            [text_section],
        )
        data_sections = [
            section_64(
                "__got",
                "__DATA",
                BASE + GOT_OFF,
                0x40,
                DATA_FILE_OFF,
            ),
            section_64(
                "__la_symbol_ptr",
                "__DATA",
                BASE + LAZY_OFF,
                0x40,
                DATA_FILE_OFF + (LAZY_OFF - DATA_VM_OFF),
            ),
            section_64(
                "__const",
                "__DATA",
                BASE + CONST_OFF,
                0x100,
                DATA_FILE_OFF + (CONST_OFF - DATA_VM_OFF),
            ),
        ]
        self.data_segment = segment_64(
            "__DATA",
            BASE + DATA_VM_OFF,
            DATA_FILE_SIZE,
            DATA_FILE_OFF,
            DATA_FILE_SIZE,
            data_sections,
        )
        self.commands.extend([self.text_segment, self.data_segment])

    def _add_default_dylibs(self):
        self.dylib_foo = "/usr/lib/libfoo.dylib"
        self.dylib_bar = "/usr/lib/libbar.dylib"
        self.commands.append(load_dylib(self.dylib_foo))
        self.commands.append(load_dylib(self.dylib_bar))

    def add_command(self, command):
        self.commands.append(command)

    def put_blob(self, blob, offset=BLOB_OFF):
        end = offset + len(blob)
        self.data[offset:end] = blob
        return offset, len(blob)

    def put_pointer(self, vm_offset, value):
        file_off = DATA_FILE_OFF + (vm_offset - DATA_VM_OFF)
        struct.pack_into("<Q", self.data, file_off, value)

    def build(self):
        sizeofcmds = sum(len(c) for c in self.commands)
        header = struct.pack(
            "<IIIIIIII",
            self.magic,
            self.cputype,
            0,
            2,  # MH_EXECUTE
            len(self.commands),
            sizeofcmds,
            0,
            0,
        )
        # self.data is the whole file image: header and commands occupy
        # its beginning and every other payload sits at its absolute offset.
        image = bytearray(self.data)
        header_size = len(header)
        image[:header_size] = header
        cursor = header_size
        for command in self.commands:
            end = cursor + len(command)
            image[cursor:end] = command
            cursor = end
        if cursor > DATA_FILE_OFF:
            raise AssertionError("load commands overflow the __TEXT segment")
        return bytes(image)


_TEMP_PATHS = []


def write_temp(content, suffix=".macho"):
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(content)
    handle.close()
    _TEMP_PATHS.append(handle.name)
    return handle.name


def load_header(content):
    path = write_temp(content)
    macho = MachO.MachO(path)
    return path, macho.headers[0]


class FixtureTestCase(unittest.TestCase):
    def tearDown(self):
        while _TEMP_PATHS:
            path = _TEMP_PATHS.pop()
            if os.path.exists(path):
                os.unlink(path)
        super(FixtureTestCase, self).tearDown()


def fat_container(slices, alignment=14):
    archs = b""
    offset = 1 << alignment
    bodies = []
    for cputype, body in slices:
        archs += struct.pack(">IIIII", cputype, 0, offset, len(body), alignment)
        bodies.append((offset, body))
        offset += len(body)
        offset = (offset + ((1 << alignment) - 1)) & ~((1 << alignment) - 1)
    header = struct.pack(">II", mach_o.FAT_MAGIC, len(slices))
    result = bytearray(offset)
    prefix_size = len(header) + len(archs)
    result[:prefix_size] = header + archs
    for slice_offset, body in bodies:
        end = slice_offset + len(body)
        result[slice_offset:end] = body
    return bytes(result)


# ---------------------------------------------------------------------------
# Chained fixups blob builder
# ---------------------------------------------------------------------------


def chained_import_word(ordinal, name_offset, weak=False):
    return (ordinal & 0xFF) | (int(weak) << 8) | (name_offset << 9)


def chain_rebase(target, next_stride):
    low = target & 0xFFFFFFFFF
    high8 = (target >> 56) & 0xFF
    return low | (high8 << 36) | (next_stride << 51)


def chain_bind(import_index, pointer_addend, next_stride):
    return (
        (1 << 63)
        | (next_stride << 51)
        | ((pointer_addend & 0xFF) << 24)
        | (import_index & 0xFFFFFF)
    )


def build_chained_blob(
    segment_entries,
    imports,
    symbols,
    imports_format=1,
    symbols_format=0,
    seg_count=None,
):
    """
    segment_entries: list of (segment_index, pointer_format, segment_offset,
                       page_size, page_starts)
    imports: list of (ordinal, name, addend)
    symbols: list of symbol name strings
    seg_count: total number of segments in the image, including segments
               without fixups whose seg_info_offset stays zero.
    """
    symbols_pool = b"".join(cstring(name) for name in symbols)
    name_offsets = {}
    running = 0
    for name in symbols:
        name_offsets[name] = running
        running += len(name) + 1

    import_bytes = bytearray()
    for ordinal, name, addend in imports:
        name_offset = name_offsets[name]
        if imports_format == 1:
            import_bytes += struct.pack("<I", chained_import_word(ordinal, name_offset))
        elif imports_format == 2:
            import_bytes += struct.pack(
                "<Ii",
                chained_import_word(ordinal, name_offset),
                addend,
            )
        else:
            word0 = (ordinal & 0xFFFF) | (name_offset << 32)
            import_bytes += struct.pack("<Qq", word0, addend)

    if seg_count is None:
        seg_count = max(index for index, _f, _o, _p, _s in segment_entries) + 1
    starts_header = struct.pack("<I", seg_count)
    seg_offsets = [0] * seg_count
    # Body is filled in once the per-segment sizes are known.
    seg_bodies = {}
    for index, pointer_format, segment_offset, page_size, starts in segment_entries:
        body = struct.pack(
            "<IHHQIH",
            22 + 2 * len(starts),
            page_size,
            pointer_format,
            segment_offset,
            0,
            len(starts),
        ) + struct.pack("<%dH" % len(starts), *starts)
        seg_bodies[index] = body

    starts_size = 4 + 4 * seg_count
    imports_offset = (
        28
        + starts_size
        + sum(len(b) for b in [seg_bodies.get(i, b"") for i in range(seg_count)])
    )
    # The seg_info_offset table is relative to the starts header; entries
    # directly follow the table.
    seg_info_bytes = b""
    cursor = 4 + 4 * seg_count
    for i in range(seg_count):
        if i in seg_bodies:
            seg_offsets[i] = cursor
            seg_info_bytes += seg_bodies[i]
            cursor += len(seg_bodies[i])
    starts_blob = (
        starts_header + struct.pack("<%dI" % seg_count, *seg_offsets) + seg_info_bytes
    )

    symbols_offset = imports_offset + len(import_bytes)
    header = struct.pack(
        "<7I",
        0,
        28,
        imports_offset,
        symbols_offset,
        len(imports),
        imports_format,
        symbols_format,
    )
    return (
        header
        + starts_blob
        + bytes(import_bytes)
        + symbols_pool
        + b"\x00" * symbols_format
    )


# ---------------------------------------------------------------------------
# Export trie builder
# ---------------------------------------------------------------------------


class TrieNode(object):
    def __init__(self, terminal=None, edges=None):
        self.terminal = terminal
        self.edges = edges or {}


def trie_terminal(kind, *values):
    return (kind,) + values


def _terminal_bytes(node):
    if node.terminal is None:
        return uleb(0)
    kind = node.terminal[0]
    if kind == "regular":
        _kind, flags, address = node.terminal
        body = uleb(flags) + uleb(address)
    elif kind == "reexport":
        _kind, ordinal, import_name = node.terminal
        body = (
            uleb(mach_o.EXPORT_SYMBOL_FLAGS_REEXPORT)
            + uleb(ordinal)
            + cstring(import_name)
        )
    elif kind == "stub":
        _kind, stub, resolver = node.terminal
        body = (
            uleb(mach_o.EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER)
            + uleb(stub)
            + uleb(resolver)
        )
    else:
        raise ValueError(kind)
    return uleb(len(body)) + body


def _serialize_node(node, base):
    terminal = _terminal_bytes(node)
    edges = sorted(node.edges.items())
    # Child offsets are absolute within the trie, while child blobs are
    # laid out directly after this node's header.  Iterate until the
    # variable uleb slot sizes agree with the offsets they encode.
    slot_sizes = [1] * len(edges)
    child_blobs = [b""] * len(edges)
    child_bases = [0] * len(edges)
    while True:
        header_len = (
            len(terminal)
            + 1
            + sum(
                len(cstring(edge)) + size
                for (edge, _child), size in zip(edges, slot_sizes)
            )
        )
        cursor = base + header_len
        for index, (_edge, child) in enumerate(edges):
            child_bases[index] = cursor
            child_blobs[index] = _serialize_node(child, cursor)
            cursor += len(child_blobs[index])
        new_sizes = [len(uleb(value)) for value in child_bases]
        if new_sizes == slot_sizes:
            break
        slot_sizes = new_sizes
    header = terminal + bytes([len(edges)])
    for (edge, _child), child_base in zip(edges, child_bases):
        header += cstring(edge) + uleb(child_base)
    return header + b"".join(child_blobs)


def serialize_trie(node):
    return _serialize_node(node, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

FIXUP_KEYS = {
    "segment",
    "section",
    "address",
    "kind",
    "target",
    "dylib",
    "dylib_ordinal",
    "symbol",
    "addend",
}
EXPORT_KEYS = {
    "name",
    "flags",
    "address",
    "reexport_dylib",
    "reexport_symbol",
}


class ClassicFixupTest(FixtureTestCase):
    def test_rebase_pointer_targets_and_sections(self):
        builder = MachOBuilder()
        builder.put_pointer(GOT_OFF, 0x100001234)
        builder.put_pointer(GOT_OFF + 8, 0x100005678)
        builder.put_pointer(0x10200, 0x100009ABC)
        stream = bytearray()
        stream += bytes(
            [
                mach_o.REBASE_OPCODE_SET_TYPE_IMM | mach_o.REBASE_TYPE_POINTER,
                mach_o.REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1,
            ]
        )
        stream += uleb(0)
        stream += bytes([mach_o.REBASE_OPCODE_DO_REBASE_IMM_TIMES | 2])
        # Move into the section-less tail of the segment before the last
        # rebase, so its section is reported as None.
        stream += bytes([mach_o.REBASE_OPCODE_ADD_ADDR_ULEB])
        stream += uleb(0x200 - 2 * 8)
        stream += bytes([mach_o.REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1])
        stream += bytes([mach_o.REBASE_OPCODE_DONE])
        off, size = builder.put_blob(bytes(stream))
        builder.add_command(dyld_info_command({"rebase_off": off, "rebase_size": size}))
        _path, header = load_header(builder.build())
        entries = header.getFixups()
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(set(e) == FIXUP_KEYS for e in entries))
        self.assertEqual(
            [e["address"] for e in entries],
            [
                BASE + GOT_OFF,
                BASE + GOT_OFF + 8,
                BASE + 0x10200,
            ],
        )
        self.assertEqual(
            [e["target"] for e in entries],
            [0x100001234, 0x100005678, 0x100009ABC],
        )
        self.assertEqual(
            [e["section"] for e in entries],
            ["__got", "__got", None],
        )
        self.assertTrue(all(e["segment"] == "__DATA" for e in entries))
        for entry in entries:
            self.assertEqual(entry["kind"], "rebase")
            self.assertIsNone(entry["dylib"])
            self.assertIsNone(entry["dylib_ordinal"])
            self.assertIsNone(entry["symbol"])
            self.assertIsNone(entry["addend"])

    def test_bind_ordinals_symbols_and_addends(self):
        builder = MachOBuilder()
        stream = bytearray()
        # Positive ordinal, plain bind with addend.
        stream += bytes([mach_o.BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | 1])
        stream += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
        stream += cstring("_foo")
        stream += bytes([mach_o.BIND_OPCODE_SET_ADDEND_SLEB])
        stream += sleb(-8)
        stream += bytes([mach_o.BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1])
        stream += uleb(0)
        stream += bytes([mach_o.BIND_OPCODE_DO_BIND])
        # ULEB ordinal 2 with scaled advance.
        stream += bytes([mach_o.BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB])
        stream += uleb(2)
        stream += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
        stream += cstring("_bar")
        stream += bytes([mach_o.BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED | 1])
        # Special ordinal -1 (main executable).
        stream += bytes([mach_o.BIND_OPCODE_SET_DYLIB_SPECIAL_IMM | 0xF])
        stream += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
        stream += cstring("_mainish")
        stream += bytes([mach_o.BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED | 1])
        # Special ordinal 0 (self).
        stream += bytes([mach_o.BIND_OPCODE_SET_DYLIB_SPECIAL_IMM])
        stream += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
        stream += cstring("_selfish")
        stream += bytes([mach_o.BIND_OPCODE_DO_BIND])
        stream += bytes([mach_o.BIND_OPCODE_DONE])
        off, size = builder.put_blob(bytes(stream))
        builder.add_command(dyld_info_command({"bind_off": off, "bind_size": size}))
        _path, header = load_header(builder.build())
        entries = [e for e in header.getFixups() if e["kind"] == "bind"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(
            [e["address"] for e in entries],
            [
                BASE + GOT_OFF,
                BASE + GOT_OFF + 8,
                BASE + GOT_OFF + 24,
                BASE + GOT_OFF + 40,
            ],
        )
        self.assertEqual([e["dylib_ordinal"] for e in entries], [1, 2, -1, 0])
        self.assertEqual(
            [e["dylib"] for e in entries],
            [
                builder.dylib_foo,
                builder.dylib_bar,
                None,
                None,
            ],
        )
        self.assertEqual(
            [e["symbol"] for e in entries],
            ["_foo", "_bar", "_mainish", "_selfish"],
        )
        self.assertEqual([e["addend"] for e in entries], [-8, -8, -8, -8])
        for entry in entries:
            self.assertIsNone(entry["target"])
            self.assertEqual(entry["section"], "__got")

    def test_bind_times_skipping_uleb(self):
        builder = MachOBuilder()
        stream = bytearray()
        stream += bytes([mach_o.BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | 1])
        stream += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
        stream += cstring("_many")
        stream += bytes([mach_o.BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1])
        stream += uleb(0)
        stream += bytes([mach_o.BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB])
        stream += uleb(3)
        stream += uleb(8)
        stream += bytes([mach_o.BIND_OPCODE_DONE])
        off, size = builder.put_blob(bytes(stream))
        builder.add_command(dyld_info_command({"bind_off": off, "bind_size": size}))
        _path, header = load_header(builder.build())
        entries = [e for e in header.getFixups() if e["kind"] == "bind"]
        self.assertEqual(
            [e["address"] for e in entries],
            [
                BASE + GOT_OFF,
                BASE + GOT_OFF + 16,
                BASE + GOT_OFF + 32,
            ],
        )
        self.assertTrue(all(e["symbol"] == "_many" for e in entries))

    def test_weak_and_lazy_streams(self):
        builder = MachOBuilder()
        weak = bytearray()
        weak += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
        weak += cstring("_weak")
        weak += bytes([mach_o.BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1])
        weak += uleb(CONST_OFF - DATA_VM_OFF)
        weak += bytes([mach_o.BIND_OPCODE_DO_BIND])
        weak += bytes([mach_o.BIND_OPCODE_DONE])

        lazy = bytearray()
        for index, name in enumerate(("_l1", "_l2")):
            lazy += bytes([mach_o.BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | 2])
            lazy += bytes([mach_o.BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM])
            lazy += cstring(name)
            lazy += bytes([mach_o.BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1])
            lazy += uleb(LAZY_OFF - DATA_VM_OFF + index * 8)
            lazy += bytes([mach_o.BIND_OPCODE_DO_BIND])
            lazy += bytes([mach_o.BIND_OPCODE_DONE])
        weak_off, weak_size = builder.put_blob(bytes(weak))
        lazy_off, lazy_size = builder.put_blob(bytes(lazy), BLOB_OFF + 0x100)
        builder.add_command(
            dyld_info_command(
                {
                    "weak_off": weak_off,
                    "weak_size": weak_size,
                    "lazy_off": lazy_off,
                    "lazy_size": lazy_size,
                }
            )
        )
        _path, header = load_header(builder.build())
        weak_entries = [e for e in header.getFixups() if e["kind"] == "weak_bind"]
        lazy_entries = [e for e in header.getFixups() if e["kind"] == "lazy_bind"]
        self.assertEqual(len(weak_entries), 1)
        self.assertEqual(weak_entries[0]["symbol"], "_weak")
        self.assertEqual(weak_entries[0]["dylib_ordinal"], 0)
        self.assertIsNone(weak_entries[0]["dylib"])
        self.assertEqual(weak_entries[0]["section"], "__const")
        self.assertEqual(len(lazy_entries), 2)
        self.assertEqual(
            [e["address"] for e in lazy_entries],
            [BASE + LAZY_OFF, BASE + LAZY_OFF + 8],
        )
        self.assertEqual([e["symbol"] for e in lazy_entries], ["_l1", "_l2"])
        self.assertTrue(all(e["dylib"] == builder.dylib_bar for e in lazy_entries))

    def test_empty_streams_return_empty_list(self):
        builder = MachOBuilder()
        builder.add_command(dyld_info_command({}))
        _path, header = load_header(builder.build())
        self.assertEqual(header.getFixups(), [])


class ChainedFixupTest(FixtureTestCase):
    def _builder_with_chain(self, pointer_format, pointers, imports, symbols):
        builder = MachOBuilder()
        page_start = GOT_OFF - DATA_VM_OFF
        for slot, value in pointers:
            builder.put_pointer(GOT_OFF + slot * 8, value)
        blob = build_chained_blob(
            [
                (
                    1,
                    pointer_format,
                    DATA_VM_OFF,
                    0x4000,
                    [page_start],
                )
            ],
            imports,
            symbols,
        )
        off, size = builder.put_blob(blob)
        builder.add_command(linkedit_command(LC_CHAINED, off, size))
        return builder

    def test_offset_format_rebase_and_bind(self):
        pointers = [
            (0, chain_rebase(0x1234, 4)),
            (
                2,
                chain_bind(0, 3, 0),
            ),
        ]
        builder = self._builder_with_chain(
            DYLD_CHAINED_PTR_64_OFFSET,
            pointers,
            [(1, "_imp1", 0)],
            ["_imp1"],
        )
        _path, header = load_header(builder.build())
        entries = header.getFixups()
        kinds = [e["kind"] for e in entries]
        self.assertEqual(kinds, ["chained_rebase", "chained_bind"])
        rebase_entry, bind_entry = entries
        self.assertEqual(rebase_entry["address"], BASE + GOT_OFF)
        self.assertEqual(rebase_entry["target"], BASE + 0x1234)
        self.assertEqual(rebase_entry["section"], "__got")
        self.assertIsNone(rebase_entry["symbol"])
        self.assertEqual(bind_entry["address"], BASE + GOT_OFF + 16)
        self.assertEqual(bind_entry["symbol"], "_imp1")
        self.assertEqual(bind_entry["dylib"], builder.dylib_foo)
        self.assertEqual(bind_entry["dylib_ordinal"], 1)
        self.assertEqual(bind_entry["addend"], 3)
        self.assertIsNone(bind_entry["target"])

    def test_plain_64_format_keeps_encoded_vmaddr(self):
        pointers = [(0, chain_rebase(0x100004321, 0))]
        builder = self._builder_with_chain(
            DYLD_CHAINED_PTR_64,
            pointers,
            [(1, "_imp1", 0)],
            ["_imp1"],
        )
        _path, header = load_header(builder.build())
        entries = header.getFixups()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["kind"], "chained_rebase")
        self.assertEqual(entries[0]["target"], 0x100004321)

    def test_import_addend_formats_and_special_ordinals(self):
        # Format 2: int32 import addend plus pointer addend.
        builder = MachOBuilder()
        blob = build_chained_blob(
            [
                (
                    1,
                    DYLD_CHAINED_PTR_64_OFFSET,
                    DATA_VM_OFF,
                    0x4000,
                    [GOT_OFF - DATA_VM_OFF],
                )
            ],
            [
                (1, "_a", -5),
                (0, "_self", 7),
                (0xF1, "_special", 0),
            ],
            ["_a", "_self", "_special"],
            imports_format=2,
        )
        off, size = builder.put_blob(blob)
        builder.add_command(linkedit_command(LC_CHAINED, off, size))
        builder.put_pointer(GOT_OFF, chain_bind(0, 4, 4))
        builder.put_pointer(GOT_OFF + 16, chain_bind(1, 1, 4))
        builder.put_pointer(GOT_OFF + 32, chain_bind(2, 0, 0))
        _path, header = load_header(builder.build())
        entries = header.getFixups()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["addend"], -5 + 4)
        self.assertEqual(entries[0]["dylib"], builder.dylib_foo)
        self.assertEqual(entries[1]["addend"], 7 + 1)
        self.assertEqual(entries[1]["dylib_ordinal"], 0)
        self.assertIsNone(entries[1]["dylib"])
        self.assertEqual(entries[2]["dylib_ordinal"], -15)
        self.assertIsNone(entries[2]["dylib"])

        # Format 3: 64-bit import entries.
        builder = MachOBuilder()
        blob = build_chained_blob(
            [
                (
                    1,
                    DYLD_CHAINED_PTR_64_OFFSET,
                    DATA_VM_OFF,
                    0x4000,
                    [GOT_OFF - DATA_VM_OFF],
                )
            ],
            [(2, "_big", 0x100)],
            ["_big"],
            imports_format=3,
        )
        off, size = builder.put_blob(blob)
        builder.add_command(linkedit_command(LC_CHAINED, off, size))
        builder.put_pointer(GOT_OFF, chain_bind(0, 0, 0))
        _path, header = load_header(builder.build())
        entries = header.getFixups()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["addend"], 0x100)
        self.assertEqual(entries[0]["dylib_ordinal"], 2)
        self.assertEqual(entries[0]["dylib"], builder.dylib_bar)

    def test_unsupported_format_and_empty_pages_are_skipped(self):
        builder = MachOBuilder()
        blob = build_chained_blob(
            [
                # Format 1 is the authenticated arm64e layout: out of scope.
                (1, 1, DATA_VM_OFF, 0x4000, [0xFFFF, 0x10]),
                (
                    2,
                    DYLD_CHAINED_PTR_64_OFFSET,
                    0x11000,
                    0x4000,
                    [0xFFFF],
                ),
            ],
            [],
            [],
        )
        off, size = builder.put_blob(blob)
        builder.add_command(linkedit_command(LC_CHAINED, off, size))
        _path, header = load_header(builder.build())
        self.assertEqual(header.getFixups(), [])

    def test_no_chained_command(self):
        builder = MachOBuilder()
        _path, header = load_header(builder.build())
        self.assertEqual(header.getFixups(), [])


class ExportTrieTest(FixtureTestCase):
    def _trie(self):
        root = TrieNode(
            edges={
                "_regular": TrieNode(terminal=trie_terminal("regular", 0, 0x100)),
                "_absolute": TrieNode(
                    terminal=trie_terminal(
                        "regular",
                        EXPORT_KIND_ABSOLUTE,
                        0xDEADBEEF,
                    )
                ),
                "_reexport_empty": TrieNode(terminal=trie_terminal("reexport", 1, "")),
                "_reexport_named": TrieNode(
                    terminal=trie_terminal("reexport", 2, "_original")
                ),
                "_stub": TrieNode(terminal=trie_terminal("stub", 0x200, 0x300)),
                "_child": TrieNode(
                    edges={
                        "_nested": TrieNode(terminal=trie_terminal("regular", 0, 0x400))
                    }
                ),
            }
        )
        return serialize_trie(root)

    def test_trie_from_dyld_info(self):
        builder = MachOBuilder()
        trie = self._trie()
        off, size = builder.put_blob(trie)
        builder.add_command(dyld_info_command({}, export_off=off, export_size=size))
        _path, header = load_header(builder.build())
        exports = header.getExports()
        self.assertTrue(all(set(e) == EXPORT_KEYS for e in exports))
        by_name = {e["name"]: e for e in exports}
        self.assertEqual(
            sorted(by_name),
            [
                "_absolute",
                "_child_nested",
                "_reexport_empty",
                "_reexport_named",
                "_regular",
                "_stub",
            ],
        )
        self.assertEqual(by_name["_regular"]["address"], 0x100)
        self.assertEqual(by_name["_regular"]["flags"], 0)
        self.assertEqual(by_name["_absolute"]["address"], 0xDEADBEEF)
        self.assertEqual(by_name["_absolute"]["flags"], EXPORT_KIND_ABSOLUTE)
        empty = by_name["_reexport_empty"]
        self.assertIsNone(empty["address"])
        self.assertEqual(empty["reexport_dylib"], builder.dylib_foo)
        self.assertEqual(empty["reexport_symbol"], "_reexport_empty")
        named = by_name["_reexport_named"]
        self.assertEqual(named["reexport_dylib"], builder.dylib_bar)
        self.assertEqual(named["reexport_symbol"], "_original")
        self.assertEqual(by_name["_stub"]["address"], 0x200)
        self.assertEqual(by_name["_child_nested"]["address"], 0x400)
        for name in ("_regular", "_absolute", "_stub", "_child_nested"):
            self.assertIsNone(by_name[name]["reexport_dylib"])
            self.assertIsNone(by_name[name]["reexport_symbol"])

    def test_trie_from_dedicated_load_command(self):
        builder = MachOBuilder()
        trie = self._trie()
        off, size = builder.put_blob(trie)
        builder.add_command(linkedit_command(LC_EXPORTS_TRIE, off, size))
        _path, header = load_header(builder.build())
        exports = header.getExports()
        self.assertEqual(len(exports), 6)

    def test_missing_trie_returns_empty(self):
        builder = MachOBuilder()
        _path, header = load_header(builder.build())
        self.assertEqual(header.getExports(), [])


class FatAndSafetyTest(FixtureTestCase):
    def test_fat_slices_keep_their_own_offsets(self):
        first = MachOBuilder(cputype=0x01000007)
        first.put_pointer(GOT_OFF, 0x11111111)
        stream = (
            bytes(
                [
                    mach_o.REBASE_OPCODE_SET_TYPE_IMM | 1,
                    mach_o.REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1,
                ]
            )
            + uleb(0)
            + bytes([mach_o.REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1])
        )
        off, size = first.put_blob(stream)
        first.add_command(dyld_info_command({"rebase_off": off, "rebase_size": size}))

        second = MachOBuilder(cputype=0x0100000C)
        second.put_pointer(GOT_OFF, 0x22222222)
        off, size = second.put_blob(stream)
        second.add_command(dyld_info_command({"rebase_off": off, "rebase_size": size}))

        fat = fat_container(
            [
                (0x01000007, first.build()),
                (0x0100000C, second.build()),
            ]
        )
        path = write_temp(fat)
        macho = MachO.MachO(path)
        self.assertEqual(len(macho.headers), 2)
        targets = [h.getFixups()[0]["target"] for h in macho.headers]
        self.assertEqual(targets, [0x11111111, 0x22222222])
        addresses = [h.getFixups()[0]["address"] for h in macho.headers]
        self.assertEqual(addresses, [BASE + GOT_OFF, BASE + GOT_OFF])

    def test_32bit_header_returns_empty_lists(self):
        header = struct.pack(
            "<IIIIIII",
            mach_o.MH_MAGIC,
            0x01000007,
            3,
            2,
            0,
            0,
            0,
        )
        path = write_temp(header)
        macho = MachO.MachO(path)
        self.assertEqual(macho.headers[0].getFixups(), [])
        self.assertEqual(macho.headers[0].getExports(), [])

    def test_reads_are_repeatable_and_do_not_mutate_the_file(self):
        builder = MachOBuilder()
        builder.put_pointer(GOT_OFF, 0x100001234)
        stream = (
            bytes(
                [
                    mach_o.REBASE_OPCODE_SET_TYPE_IMM | 1,
                    mach_o.REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1,
                ]
            )
            + uleb(0)
            + bytes([mach_o.REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1])
        )
        off, size = builder.put_blob(stream)
        builder.add_command(dyld_info_command({"rebase_off": off, "rebase_size": size}))
        trie = serialize_trie(
            TrieNode(edges={"_x": TrieNode(terminal=trie_terminal("regular", 0, 0x10))})
        )
        trie_off, trie_size = builder.put_blob(trie, BLOB_OFF + 0x100)
        builder.add_command(linkedit_command(LC_EXPORTS_TRIE, trie_off, trie_size))
        content = builder.build()
        digest_before = hashlib.sha256(content).hexdigest()
        path = write_temp(content)
        macho = MachO.MachO(path)
        header = macho.headers[0]
        commands_before = len(header.commands)
        fixups_one = header.getFixups()
        fixups_two = header.getFixups()
        exports_one = header.getExports()
        exports_two = header.getExports()
        self.assertIs(fixups_one, fixups_two)
        self.assertIs(exports_one, exports_two)
        self.assertEqual(len(header.commands), commands_before)
        self.assertEqual(
            hashlib.sha256(open(path, "rb").read()).hexdigest(),
            digest_before,
        )

    def test_write_flow_unaffected_by_fixup_reads(self):
        builder = MachOBuilder()
        stream = bytes([mach_o.REBASE_OPCODE_DONE])
        off, size = builder.put_blob(stream)
        builder.add_command(dyld_info_command({"rebase_off": off, "rebase_size": size}))
        content = builder.build()

        path_one = write_temp(content)
        header_one = MachO.MachO(path_one).headers[0]
        output_one = tempfile.NamedTemporaryFile(delete=False)
        header_one.write(output_one)
        output_one.close()

        path_two = write_temp(content)
        header_two = MachO.MachO(path_two).headers[0]
        header_two.getFixups()
        header_two.getExports()
        output_two = tempfile.NamedTemporaryFile(delete=False)
        header_two.write(output_two)
        output_two.close()

        written_one = open(output_one.name, "rb").read()
        written_two = open(output_two.name, "rb").read()
        self.assertEqual(written_one, written_two)
        # The serialized header region still carries the dependency names
        # and the dyld info load command used by the rewrite flow.
        self.assertIn(builder.dylib_foo.encode(), written_two)
        self.assertIn(builder.dylib_bar.encode(), written_two)
        # The in-memory dependency table is untouched by the read calls.
        names = [value for _idx, _name, value in header_two.walkRelocatables()]
        self.assertEqual(names, [builder.dylib_foo, builder.dylib_bar])
        os.unlink(output_one.name)
        os.unlink(output_two.name)


@unittest.skipUnless(os.path.exists("/bin/ls"), "system sample binary unavailable")
class RealBinarySmokeTest(unittest.TestCase):
    def test_system_fat_binary(self):
        macho = MachO.MachO("/bin/ls")
        self.assertGreaterEqual(len(macho.headers), 1)
        for header in macho.headers:
            fixups = header.getFixups()
            exports = header.getExports()
            self.assertTrue(all(set(entry) == FIXUP_KEYS for entry in fixups))
            self.assertTrue(all(set(entry) == EXPORT_KEYS for entry in exports))
            for entry in fixups:
                self.assertIsInstance(entry["address"], int)
                if entry["kind"] in ("rebase", "chained_rebase"):
                    self.assertIsNotNone(entry["target"])
                    self.assertIsNone(entry["symbol"])
                else:
                    self.assertIsNone(entry["target"])
                    self.assertIsNotNone(entry["symbol"])
                    if entry["dylib_ordinal"] and entry["dylib_ordinal"] > 0:
                        self.assertTrue(entry["dylib"].endswith(".dylib"))

    @unittest.skipUnless(
        os.path.exists("/opt/homebrew/bin/adig"),
        "chained fixups sample unavailable",
    )
    def test_chained_fixups_sample(self):
        macho = MachO.MachO("/opt/homebrew/bin/adig")
        header = macho.headers[0]
        fixups = header.getFixups()
        kinds = set(entry["kind"] for entry in fixups)
        self.assertIn("chained_rebase", kinds)
        self.assertIn("chained_bind", kinds)
        for entry in fixups:
            self.assertEqual(entry["segment"], "__DATA_CONST")


if __name__ == "__main__":
    unittest.main()

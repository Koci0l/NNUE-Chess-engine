#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


def write_embed_header(src: Path, dst: Path, name: str) -> None:
    data = src.read_bytes()

    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#pragma once\n")
        f.write("#include <cstddef>\n")
        f.write("#include <cstdint>\n")
        f.write("\n")
        f.write(f"// Auto-generated from {src.name} ({len(data)} bytes). Do not edit by hand.\n")
        f.write("\n")

        f.write(f"inline constexpr std::size_t {name}_size = {len(data)};\n")
        f.write("\n")

        f.write(f"alignas(64) inline const std::uint8_t {name}_data[] = {{\n")

        if len(data) == 0:
            f.write("    0x00\n")
        else:
            bytes_per_line = 16

            for i in range(0, len(data), bytes_per_line):
                chunk = data[i:i + bytes_per_line]
                line = "    " + ", ".join(f"0x{b:02x}" for b in chunk) + ","
                f.write(line + "\n")

        f.write("};\n")

    print(f"wrote {dst} from {src} ({len(data)} bytes, prefix={name})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed a binary policy net as a C++ header."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input .bin policy net file"
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output .h header file"
    )

    parser.add_argument(
        "name_pos",
        nargs="?",
        default=None,
        help="C++ symbol prefix, e.g. g_policy_embed or g_policy_small_embed"
    )

    parser.add_argument(
        "--name",
        dest="name_opt",
        default=None,
        help="C++ symbol prefix, e.g. g_policy_embed or g_policy_small_embed"
    )

    args = parser.parse_args()

    name = args.name_opt or args.name_pos or "g_policy_embed"

    if not name.isidentifier():
        print(f"error: invalid C++ symbol prefix: {name}", file=sys.stderr)
        sys.exit(1)

    src = args.input
    dst = args.output

    if not src.is_file():
        print(f"error: missing {src}", file=sys.stderr)
        sys.exit(1)

    write_embed_header(src, dst, name)


if __name__ == "__main__":
    main()
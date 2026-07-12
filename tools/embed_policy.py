#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("usage: embed_policy.py quantised.bin src/policy_embed.h")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.is_file():
        print(f"error: missing {src}")
        sys.exit(1)

    data = src.read_bytes()
    lines = [
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        f"// Auto-generated from {src.name} ({len(data)} bytes). Do not edit by hand.",
        f"static constexpr std::size_t g_policy_embed_size = {len(data)};",
        "static const std::uint8_t g_policy_embed_data[] = {",
    ]

    row = []
    for b in data:
        row.append(f"0x{b:02x}")
        if len(row) == 16:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")

    lines.append("};")
    lines.append("")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {dst} ({len(data)} bytes)")

if __name__ == "__main__":
    main()
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

check=0
args=(-i)
if [[ "${1:-}" == "--check" ]]; then
    check=1
    args=(--dry-run -Werror)
fi

status=0
find src include -type f \( -name '*.cpp' -o -name '*.h' \) \
    -not -name 'stb_*.h' -print0 |
    xargs -0 clang-format-20 "${args[@]}" || status=1

format_slang() {
    local formatted
    formatted=$(clang-format-20 --assume-filename="$1.cs" <"$1")
    if [[ "$check" == 1 ]]; then
        printf '%s\n' "$formatted" | diff -u --label "$1" "$1" -
    elif ! printf '%s\n' "$formatted" | cmp -s - "$1"; then
        printf '%s\n' "$formatted" >"$1"
    fi
}
export -f format_slang
export check
find shaders -type f -name '*.slang' -print0 |
    xargs -0 -n1 bash -c 'format_slang "$1"' _ || status=1

exit $status

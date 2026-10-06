#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
umc_root="$project_root/third_party/umc"
timing_patch="$project_root/third_party/umc_timing.patch"
if ! git -C "$umc_root" diff --quiet -- include/compress.hpp include/decompress.hpp;then
    echo "UMC源码已有修改；为避免覆盖这些修改，停止构建" >&2
    exit 1
fi
umc_backup_dir="$(mktemp -d)"
cp "$umc_root/include/compress.hpp" "$umc_backup_dir/compress.hpp"
cp "$umc_root/include/decompress.hpp" "$umc_backup_dir/decompress.hpp"

restore_official_source()
{
    cp "$umc_backup_dir/compress.hpp" "$umc_root/include/compress.hpp"
    cp "$umc_backup_dir/decompress.hpp" "$umc_root/include/decompress.hpp"
    rm -r -- "$umc_backup_dir"
}
trap restore_official_source EXIT

if git -C "$umc_root" apply --ignore-space-change --ignore-whitespace --reverse --check "$timing_patch" >/dev/null 2>&1;then
    :
elif git -C "$umc_root" apply --ignore-space-change --ignore-whitespace --check "$timing_patch" >/dev/null 2>&1;then
    git -C "$umc_root" apply --ignore-space-change --ignore-whitespace "$timing_patch"
else
    echo "UMC计时补丁既不能应用，也不是已应用状态" >&2
    exit 1
fi

if [[ ! -d "$umc_root/external/SZ3/install" ]];then
    echo "UMC依赖尚未安装，请先在third_party/umc中执行官方build_script.sh" >&2
    exit 1
fi

cmake -S "$umc_root" -B "$umc_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$umc_root/build" --target test_compress test_decompress --parallel "${1:-8}"

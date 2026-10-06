#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
umc_root="$project_root/third_party/umc"
replacement_root="$project_root/third_party/umc_replacements"
upstream_url="https://github.com/xuanwu02/umc.git"
upstream_commit="52a86589de096cadfd8ab83719dc4c3e697e5886"
replacement_paths=(include/compress.hpp include/decompress.hpp)

if [[ ! -d "$umc_root/.git" && ! -f "$umc_root/.git" ]];then
    echo "没有找到UMC子模块，请先执行git submodule update --init third_party/umc" >&2
    exit 1
fi
if [[ "$(git -C "$umc_root" remote get-url origin)" != "$upstream_url" ]];then
    echo "UMC origin不是预期的官方仓库：$(git -C "$umc_root" remote get-url origin)" >&2
    exit 1
fi
git -C "$umc_root" fetch --quiet origin main
if ! git -C "$umc_root" cat-file -e "$upstream_commit^{commit}";then
    echo "官方UMC中没有找到固定commit：$upstream_commit" >&2
    exit 1
fi
if [[ "$(git -C "$umc_root" rev-parse HEAD)" != "$upstream_commit" ]];then
    echo "UMC没有处于固定commit：$upstream_commit" >&2
    exit 1
fi

mapfile -t changed_paths < <(git -C "$umc_root" diff --name-only "$upstream_commit" --)
if (( ${#changed_paths[@]}==0 ));then
    for path in "${replacement_paths[@]}";do
        cp "$replacement_root/$path" "$umc_root/$path"
    done
    echo "已安装UNSZ保存的UMC计时替换文件"
else
    if (( ${#changed_paths[@]}!=${#replacement_paths[@]} ));then
        printf 'UMC包含未知源码修改，停止构建：\n%s\n' "${changed_paths[*]}" >&2
        exit 1
    fi
    for path in "${replacement_paths[@]}";do
        if ! printf '%s\n' "${changed_paths[@]}" | grep -Fxq "$path"||! cmp -s "$replacement_root/$path" "$umc_root/$path";then
            echo "UMC包含未知源码修改，停止构建：$path" >&2
            exit 1
        fi
    done
    echo "UMC计时替换文件已经安装"
fi

if [[ ! -d "$umc_root/external/SZ3/install" ]];then
    echo "UMC依赖尚未安装，请先在third_party/umc中执行官方build_script.sh" >&2
    exit 1
fi

cmake -S "$umc_root" -B "$umc_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$umc_root/build" --target test_compress test_decompress --parallel "${1:-8}"

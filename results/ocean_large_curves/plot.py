import json
from pathlib import Path

import matplotlib.pyplot as plt


result_dir=Path(__file__).resolve().parent
figure_dir=result_dir/"figures"
figure_dir.mkdir(parents=True,exist_ok=True)
results=[json.loads(path.read_text()) for path in sorted(result_dir.glob("metrics_*.json"))]
if len(results)!=5:
    raise RuntimeError(f"需要5个EB结果，实际为{len(results)}")
styles={"unsz":{"label":"UNSZ","color":"#0072B2","marker":"s"},"umc":{"label":"UMC","color":"#D55E00","marker":"o"}}


def save(figure,name):
    figure.tight_layout()
    figure.savefig(figure_dir/(name+".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(figure_dir/(name+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)


figure,axis=plt.subplots(figsize=(5.2,3.8))
for compressor,style in styles.items():
    curve=sorted(results,key=lambda item:item[compressor]["bit_rate"])
    axis.plot([item[compressor]["bit_rate"] for item in curve],[item[compressor]["normalized_psnr"] for item in curve],label=style["label"],color=style["color"],marker=style["marker"],linewidth=2)
axis.set_xscale("log")
axis.set_xlabel("Bit rate (bit/value, log scale)")
axis.set_ylabel("Normalized PSNR (dB)")
axis.grid(True,linestyle=":",alpha=.55)
axis.legend(frameon=False)
figure.text(.5,-.02,"Ocean-large: rate-distortion",ha="center",va="top",fontsize=12)
save(figure,"ocean_large_rd")

figure,axis=plt.subplots(figsize=(5.2,3.8))
for compressor,style in styles.items():
    curve=sorted(results,key=lambda item:item["rel_eb"])
    x=[item["rel_eb"] for item in curve]
    axis.plot(x,[item[compressor]["compression_MB_s"] for item in curve],label=style["label"]+" compression",color=style["color"],marker=style["marker"],linewidth=2)
    axis.plot(x,[item[compressor]["decompression_MB_s"] for item in curve],label=style["label"]+" decompression",color=style["color"],marker=style["marker"],linewidth=2,linestyle="--")
axis.set_xscale("log")
axis.set_ylim(bottom=0)
axis.set_xlabel("REL EB (log scale)")
axis.set_ylabel("Core throughput (MB/s)")
axis.grid(True,which="both",linestyle=":",alpha=.55)
axis.legend(frameon=False,ncol=2,fontsize=8)
figure.text(.5,-.02,"Ocean-large: core throughput",ha="center",va="top",fontsize=12)
save(figure,"ocean_large_releb_throughput")

selected=next(item for item in results if abs(item["rel_eb"]-1e-3)<1e-12)
components=(("compression_preprocessing_time","Compression preprocessing","#56B4E9"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))
figure,axis=plt.subplots(figsize=(4.6,3.8))
for index,compressor in enumerate(("unsz","umc")):
    bottom=0
    for key,label,color in components:
        value=selected[compressor][key]
        axis.bar(index,value,bottom=bottom,color=color,edgecolor="black",linewidth=.35,label=label if index==0 else None)
        bottom+=value
axis.set_xticks((0,1),("UNSZ","UMC"))
axis.set_ylabel("Time (s)")
axis.grid(True,axis="y",linestyle=":",alpha=.55)
axis.legend(frameon=False,fontsize=8)
figure.text(.5,-.02,"Ocean-large: REL EB=1e-3 timing",ha="center",va="top",fontsize=12)
save(figure,"ocean_large_releb_1e-3_time")

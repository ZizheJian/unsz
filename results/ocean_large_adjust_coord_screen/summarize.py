import csv,sys
from pathlib import Path


project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project/"py_codes"))
from parse_unsz_output import parse_unsz_output

result_dir=Path(__file__).resolve().parent
fields=(result_dir/"fields.txt").read_text().splitlines()
rows=[]
for field in fields:
    by_mode={}
    for mode in ("on","off"):
        compression_output=(result_dir/"logs"/f"{field}_{mode}_compress.log").read_text()
        decompression_output=(result_dir/"logs"/f"{field}_{mode}_decompress.log").read_text()
        by_mode[mode]=parse_unsz_output(compression_output,decompression_output)
    on,off=by_mode["on"],by_mode["off"]
    rows.append({"field":field,"on_br":on["br"],"off_br":off["br"],"off_over_on_br":off["br"]/on["br"],"on_psnr":on["psnr"],"off_psnr":off["psnr"],"on_coordinate_transform_time":on["coordinate_transform_time"],"off_coordinate_transform_time":off["coordinate_transform_time"],"on_prediction_structure_time":on["prediction_structure_time"],"off_prediction_structure_time":off["prediction_structure_time"],"on_automatic_flow_selection_time":on["automatic_flow_selection_time"],"off_automatic_flow_selection_time":off["automatic_flow_selection_time"],"on_compression_time":on["compression_time"],"off_compression_time":off["compression_time"],"on_decompression_time":on["decompression_time"],"off_decompression_time":off["decompression_time"],"on_label_flow":on["selected_label_flow"],"on_value_flow":on["selected_value_flow"],"off_label_flow":off["selected_label_flow"],"off_value_flow":off["selected_value_flow"]})
writer=csv.DictWriter(sys.stdout,fieldnames=rows[0].keys())
writer.writeheader()
writer.writerows(rows)

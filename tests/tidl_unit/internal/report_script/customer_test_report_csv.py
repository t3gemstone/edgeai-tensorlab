#!/usr/bin/env python3
import os
import csv
from bs4 import BeautifulSoup
import sys
import argparse
import shutil

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('--reports_path', help='Path to pytest html test reports', type=str, required=True)
args = parser.parse_args()

reports_dir     = args.reports_path
configs_dir  = "../../tidl_unit_test_data/configs"
out_dir      = "customer_test_reports"
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

def parse_html_report(html_path):
    """
    Returns a dict mapping model_name -> (status, subgraphs, offload)
    """
    results = {}
    if not os.path.exists(html_path):
        return results

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    rows = soup.select("table#results-table > tbody.results-table-row")
    for row in rows:
        status    = row.select_one("td.col-result").get_text(strip=True)
        test_cell = row.select_one("td.col-name").get_text(strip=True)
        subgraphs = row.select_one("td.col-name + td").get_text(strip=True)
        offload = row.select_one("td.col-name + td + td").get_text(strip=True)
        if offload == "-":
            offload = "False"
        if status.lower() == "error" or status.lower() == "xfailed":
            status = "Failed"

        # extract the bracketed model name
        if "[" in test_cell and "]" in test_cell:
            mn = test_cell.split("[",1)[1].split("]",1)[0]
        else:
            mn = test_cell

        results[mn] = (status, subgraphs, offload)
    return results

VARIANTS = [
    ("infer_ref_with_nc",      "Inference Host"),
    ("infer_target_with_nc",   "Inference Target"),
]

operator_summaries = []

for soc in sorted(os.listdir(reports_dir)):
    soc_dir = os.path.join(reports_dir, soc)

    for op in sorted(os.listdir(soc_dir)):
        op_dir = os.path.join(soc_dir, op)

        mod_num = None
        if '_' in op:
            mod_num = op.split('_')[-1]
            op = "_".join(op.split('_')[:-1])

        data = {k: parse_html_report(os.path.join(op_dir, f"{k}.html")) for k,_ in VARIANTS}

        # collect all model names from parsed HTML reports
        models = set().union(*[d.keys() for d in data.values()], *[])

        out_path = os.path.join(out_dir, f"{op}.csv")
        header = ["Model name"]
        header += ["TIDL_Offload"]
        for _, label in VARIANTS:
            col = label.replace(" ", "_")
            header += [f"{col}"]

        op_cfg_dir = os.path.join(configs_dir, op)

        model_attrs = {}
        all_attr_keys = set()

        # Only read configs if directory exists
        if os.path.isdir(op_cfg_dir):
            for fn in os.listdir(op_cfg_dir):
                if not fn.endswith(".csv"):
                    continue
                path = os.path.join(op_cfg_dir, fn)
                with open(path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        mn = row["model name"]
                        if mod_num and f"{op}_{mod_num}" != mn:
                            continue
                        if mn not in model_attrs:
                            model_attrs[mn] = {}
                        for k,v in row.items():
                            if k == "model name":
                                continue
                            model_attrs[mn][k] = v
                            all_attr_keys.add(k)

            header += sorted(all_attr_keys)
        else:
            # config dir missing - skip config columns
            model_attrs = {}
            all_attr_keys = set()

        num_offload = 0
        total_tests = 0

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            # If configs present, use models from model_attrs, else from parsed data
            if model_attrs:
                model_list = sorted(model_attrs.keys())
            else:
                model_list = sorted(models)

            for mn in model_list:
                attrs = model_attrs.get(mn, {})
                total_tests += 1
                row = [mn]

                # Find offload status from any variant available
                offload_val = "-"
                for key, _ in VARIANTS:
                    if mn in data[key]:
                        offload_val = data[key][mn][2].lower()
                        break

                if offload_val == "true":
                    num_offload += 1
                    row.append("true")
                else:
                    row.append("false")

                for key, _ in VARIANTS:
                    sa, subg, offload = data[key].get(mn, ("N/A", "", ""))
                    row.append(sa)

                for k in sorted(all_attr_keys):
                    row.append(attrs.get(k, ""))

                w.writerow(row)

        op_summary = {"Operator": op, "TIDL_Offload_Percentage": num_offload, "Total_Tests": total_tests}
        total_val = 0
        for key, _ in VARIANTS:
            da = data[key]
            if da:
                p = sum(1 for s,__,___ in da.values() if s.lower() == "passed")
                f = sum(1 for s,__,___ in da.values() if s.lower() != "passed")
                op_summary["Inference_test_result_passed"] = str(p)
                total_val = p + f
            else:
                op_summary["Inference_test_result_passed"] = "-"

        # Avoid division by zero
        if total_val > 0:
            op_summary["TIDL_Offload_Percentage"] = (num_offload / total_val) * 100
        else:
            op_summary["TIDL_Offload_Percentage"] = 0

        operator_summaries.append(op_summary)
        print(f"Wrote {out_path}")

# --- FULL OPERATOR COMPARISON ---
full_path = os.path.join(out_dir, "operator_test_report_summary.csv")
fields = ["Operator", "TIDL_Offload_Percentage", "Total_Tests", "Inference_test_result_passed"]

with open(full_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rec in sorted(operator_summaries, key=lambda x: x["Operator"]):
        w.writerow(rec)

print(f"Wrote consolidated comparison → {full_path}")
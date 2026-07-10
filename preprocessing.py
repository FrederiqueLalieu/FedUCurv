import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from sklearn.model_selection import train_test_split


BLOCK_EVENTS = {"Remove Payment Block", "Set Payment Block"}
ITEM_BLOCK_EVENT = "Block Purchase Order Item"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="BPI_Challenge_2019.xes",
        help="path to the raw bpi xes file",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="where the raw case tables and audit should go",
    )
    parser.add_argument(
        "--train-prop",
        type=float,
        default=0.8,
        help="share of cases that goes to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the train test split",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="optional cap for quick smoke runs",
    )
    return parser.parse_args()


def parse_bool(value):
    if value is None:
        return 0
    return int(str(value).lower() == "true")


def parse_time(value):
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds_between(start, end):
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def compute_case_row(trace):
    trace_attrs = {}
    n_events = 0
    n_invoices = 0
    n_goods_receipts = 0
    n_clearing_events = 0
    blocked_event_count = 0
    item_block_event_count = 0
    order_value = 0.0
    sum_invoices = 0.0

    case_start = None
    case_end = None
    first_goods_receipt = None
    first_invoice_receipt = None
    first_clear_invoice = None

    for child in trace:
        if child.tag != "event":
            trace_attrs[child.get("key")] = child.get("value")
            continue

        n_events += 1
        event_attrs = {field.get("key"): field.get("value") for field in child}
        event_name = event_attrs.get("concept:name")
        event_time = parse_time(event_attrs.get("time:timestamp"))
        event_value = event_attrs.get("Cumulative net worth (EUR)")

        if event_time is not None:
            if case_start is None or event_time < case_start:
                case_start = event_time
            if case_end is None or event_time > case_end:
                case_end = event_time

        if event_value is not None:
            value = float(event_value)
            if value > order_value:
                order_value = value
            if event_name == "Record Invoice Receipt":
                sum_invoices += value

        if event_name == "Record Invoice Receipt":
            n_invoices += 1
            if first_invoice_receipt is None:
                first_invoice_receipt = event_time
        elif event_name == "Record Goods Receipt":
            n_goods_receipts += 1
            if first_goods_receipt is None:
                first_goods_receipt = event_time
        elif event_name == "Clear Invoice":
            n_clearing_events += 1
            if first_clear_invoice is None:
                first_clear_invoice = event_time

        if event_name in BLOCK_EVENTS:
            blocked_event_count += 1
        elif event_name == ITEM_BLOCK_EVENT:
            item_block_event_count += 1

    goods_till_invoice = seconds_between(first_goods_receipt, first_invoice_receipt)
    goods_till_clear = seconds_between(first_goods_receipt, first_clear_invoice)

    has_invoice_value = int(sum_invoices > 0)
    if has_invoice_value:
        log_ratio = float(np.log1p(order_value / sum_invoices))
    else:
        log_ratio = 0.0

    return {
        "case_id": trace_attrs.get("concept:name"),
        "Purchasing Document": trace_attrs.get("Purchasing Document"),
        "Item": trace_attrs.get("Item"),
        "Vendor": trace_attrs.get("Vendor"),
        "Name": trace_attrs.get("Name"),
        "Company": trace_attrs.get("Company"),
        "Item Type": trace_attrs.get("Item Type"),
        "Document Type": trace_attrs.get("Document Type"),
        "Spend area text": trace_attrs.get("Spend area text"),
        "Sub spend area text": trace_attrs.get("Sub spend area text"),
        "Source": trace_attrs.get("Source"),
        "Item Category": trace_attrs.get("Item Category"),
        "GR-Based Inv. Verif.": parse_bool(trace_attrs.get("GR-Based Inv. Verif.")),
        "Goods Receipt": parse_bool(trace_attrs.get("Goods Receipt")),
        "n_events": n_events,
        "n_invoices": n_invoices,
        "n_goods_receipts": n_goods_receipts,
        "n_clearing_events": n_clearing_events,
        "order_value": order_value,
        "sum_invoices": sum_invoices,
        "log_ratio": log_ratio,
        "case_duration": seconds_between(case_start, case_end) or 0.0,
        "goods_till_invoice_missing": int(goods_till_invoice is None),
        "goods_till_invoice": goods_till_invoice or 0.0,
        "goods_till_clear_missing": int(goods_till_clear is None),
        "goods_till_clear": goods_till_clear or 0.0,
        "has_goods_receipt": int(first_goods_receipt is not None),
        "has_invoice_receipt": int(first_invoice_receipt is not None),
        "has_clear_invoice": int(first_clear_invoice is not None),
        "blocked_event_count": blocked_event_count,
        "item_block_event_count": item_block_event_count,
        "was_blocked": int(blocked_event_count > 0),
    }


def build_case_table(xes_path, max_traces=None):
    rows = []
    context = etree.iterparse(str(xes_path), events=("end",), tag="trace")

    for trace_idx, (_, trace) in enumerate(context, start=1):
        rows.append(compute_case_row(trace))

        # keep memory flat on the full file
        trace.clear()
        while trace.getprevious() is not None:
            del trace.getparent()[0]

        if max_traces is not None and trace_idx >= max_traces:
            break

    return pd.DataFrame(rows)


def split_case_table(case_table, train_prop, seed):
    stratify = None
    if case_table["was_blocked"].nunique() > 1:
        stratify = case_table["was_blocked"]

    train_df, test_df = train_test_split(
        case_table,
        train_size=train_prop,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    return (
        train_df.sort_values("case_id").reset_index(drop=True),
        test_df.sort_values("case_id").reset_index(drop=True),
    )


def build_audit(case_table, train_df, test_df):
    return {
        "n_cases": int(len(case_table)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "blocked_cases": int(case_table["was_blocked"].sum()),
        "item_block_cases": int((case_table["item_block_event_count"] > 0).sum()),
        "blocked_event_total": int(case_table["blocked_event_count"].sum()),
        "item_block_event_total": int(case_table["item_block_event_count"].sum()),
        "blocked_rate": float(case_table["was_blocked"].mean()),
        "train_blocked_rate": float(train_df["was_blocked"].mean()),
        "test_blocked_rate": float(test_df["was_blocked"].mean()),
        "missing_goods_till_invoice": int(case_table["goods_till_invoice_missing"].sum()),
        "missing_goods_till_clear": int(case_table["goods_till_clear_missing"].sum()),
        "top_companies": case_table["Company"].value_counts().head(10).to_dict(),
        "top_spend_areas": case_table["Spend area text"].value_counts().head(10).to_dict(),
    }


def write_outputs(case_table, train_df, test_df, audit, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    case_table.to_pickle(output_dir / "raw_data.pkl")
    train_df.to_pickle(output_dir / "raw_train_data.pkl")
    test_df.to_pickle(output_dir / "raw_test_data.pkl")
    with open(output_dir / "preprocessing_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)


def main():
    args = parse_args()
    xes_path = Path(args.input)
    output_dir = Path(args.output_dir)

    case_table = build_case_table(xes_path, max_traces=args.max_traces)
    train_df, test_df = split_case_table(case_table, args.train_prop, args.seed)
    audit = build_audit(case_table, train_df, test_df)
    write_outputs(case_table, train_df, test_df, audit, output_dir)

    print(f"wrote {len(case_table)} cases to {output_dir}")
    print(f"blocked rate: {audit['blocked_rate']:.4f}")


if __name__ == "__main__":
    main()

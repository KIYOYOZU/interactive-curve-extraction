from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from interactive_extract_second_plot import Calibration, export_to_excel, load_calibration_from_dat


POINT_PATTERN = re.compile(
    r"^\[(?P<label>黑色实验数据|蓝色曲线|红色曲线)\]\s+"
    r"第\s+(?P<point_index>\d+)/(?P<point_total>\d+)\s+个点"
    r"（垂线\s+(?P<line_index>\d+)/(?P<line_total>\d+)）："
    r"x\s+=\s+(?P<x>-?\d+(?:\.\d+)?),\s+y\s+=\s+(?P<y>-?\d+(?:\.\d+)?)$"
)
UNDO_PATTERN = re.compile(r"^\[(?P<label>黑色实验数据|蓝色曲线|红色曲线)\]\s+已撤销上一点")

LABEL_TO_KEY = {
    "黑色实验数据": "black",
    "蓝色曲线": "blue",
    "红色曲线": "red",
}


def resolve_path_from_args(default_name: str, arg_index: int) -> Path:
    if len(sys.argv) > arg_index:
        return Path(sys.argv[arg_index]).expanduser().resolve()
    return (Path.cwd() / default_name).resolve()


def build_calibration_frame(calibration: Calibration) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label": "x_start",
                "pixel_x": calibration.x_start_px[0],
                "pixel_y": calibration.x_start_px[1],
                "axis_value": calibration.x_start_val,
            },
            {
                "label": "x_end",
                "pixel_x": calibration.x_end_px[0],
                "pixel_y": calibration.x_end_px[1],
                "axis_value": calibration.x_end_val,
            },
            {
                "label": "y_start",
                "pixel_x": calibration.y_start_px[0],
                "pixel_y": calibration.y_start_px[1],
                "axis_value": calibration.y_start_val,
            },
            {
                "label": "y_end",
                "pixel_x": calibration.y_end_px[0],
                "pixel_y": calibration.y_end_px[1],
                "axis_value": calibration.y_end_val,
            },
        ]
    )


def parse_log_file(log_path: Path) -> tuple[dict[str, list[tuple[int, float, float]]], int]:
    curve_events: dict[str, list[tuple[int, float, float]]] = {key: [] for key in LABEL_TO_KEY.values()}
    line_total: int | None = None

    with log_path.open("r", encoding="utf-8") as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue

            point_match = POINT_PATTERN.match(line)
            if point_match:
                label = point_match.group("label")
                key = LABEL_TO_KEY[label]
                current_line_total = int(point_match.group("line_total"))
                if line_total is None:
                    line_total = current_line_total
                elif line_total != current_line_total:
                    raise ValueError(f"日志中的总垂线数不一致：{line_total} 与 {current_line_total}")

                curve_events[key].append(
                    (
                        int(point_match.group("line_index")),
                        float(point_match.group("x")),
                        float(point_match.group("y")),
                    )
                )
                continue

            undo_match = UNDO_PATTERN.match(line)
            if undo_match:
                label = undo_match.group("label")
                key = LABEL_TO_KEY[label]
                if not curve_events[key]:
                    raise ValueError(f"日志中的撤销操作无法匹配已有点：{line}")
                curve_events[key].pop()

    if line_total is None:
        raise ValueError("日志中未找到任何有效的数据点记录。")

    return curve_events, line_total


def build_frames_from_logs(
    curve_events: dict[str, list[tuple[int, float, float]]],
    line_total: int,
    calibration: Calibration | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calibration is not None:
        x_values = np.linspace(calibration.x_start_val, calibration.x_end_val, line_total)
    else:
        x_values = np.full(line_total, np.nan, dtype=float)
        for events in curve_events.values():
            for line_index, x_value, _ in events:
                x_values[line_index - 1] = x_value

    data = {
        "black": [np.nan] * line_total,
        "blue": [np.nan] * line_total,
        "red": [np.nan] * line_total,
    }

    for key, events in curve_events.items():
        for line_index, _x_value, y_value in events:
            data[key][line_index - 1] = y_value

    data_frame = pd.DataFrame(data)
    x_frame = pd.DataFrame({"x": x_values})
    return data_frame, x_frame


def export_recovered_excel(
    output_path: Path,
    data_frame: pd.DataFrame,
    x_frame: pd.DataFrame,
    calibration: Calibration | None,
    log_path: Path,
) -> None:
    export_to_excel(output_path, data_frame, x_frame)

    if not output_path.exists():
        raise FileNotFoundError(f"Excel 导出失败：{output_path}")

    if calibration is None or importlib.util.find_spec("openpyxl") is None:
        return

    calibration_frame = build_calibration_frame(calibration)
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        calibration_frame.to_excel(writer, sheet_name="calibration", index=False)
        pd.DataFrame({"source_log": [str(log_path)]}).to_excel(writer, sheet_name="meta", index=False)


def main() -> None:
    log_path = resolve_path_from_args("logs.txt", 1)
    dat_path = resolve_path_from_args("图片1_提取结果.dat", 2)
    output_path = resolve_path_from_args("图片1_提取结果.xlsx", 3)

    if not log_path.exists():
        raise FileNotFoundError(f"日志文件不存在：{log_path}")

    calibration = load_calibration_from_dat(dat_path)
    curve_events, line_total = parse_log_file(log_path)
    data_frame, x_frame = build_frames_from_logs(curve_events, line_total, calibration)
    export_recovered_excel(output_path, data_frame, x_frame, calibration, log_path)

    print(f"日志已恢复为 Excel：{output_path}")
    print(f"总垂线数：{line_total}")
    print(f"black 有效点数：{int(data_frame['black'].notna().sum())}")
    print(f"blue 有效点数：{int(data_frame['blue'].notna().sum())}")
    print(f"red 有效点数：{int(data_frame['red'].notna().sum())}")


if __name__ == "__main__":
    main()

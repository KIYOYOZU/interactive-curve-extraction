from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from interactive_extract_second_plot import Calibration, load_calibration_from_dat


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


def resolve_path(default_name: str, arg_index: int) -> Path:
    if len(sys.argv) > arg_index:
        return Path(sys.argv[arg_index]).expanduser().resolve()
    return (Path.cwd() / default_name).resolve()


def load_excel_data(excel_path: Path) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在：{excel_path}")

    data_frame = pd.read_excel(excel_path, sheet_name="data")
    required_columns = {"x", "black", "blue", "red"}
    missing = required_columns - set(data_frame.columns)
    if missing:
        raise ValueError(f"Excel 的 `data` 工作表缺少列：{sorted(missing)}")
    return data_frame


def compute_step2_view(calibration: Calibration) -> tuple[float, float, float, float]:
    crop_x_min = min(calibration.x_start_px[0], calibration.x_end_px[0])
    crop_x_max = max(calibration.x_start_px[0], calibration.x_end_px[0])

    y_low_val = min(calibration.y_start_val, calibration.y_end_val)
    y_high_val = max(calibration.y_start_val, calibration.y_end_val)
    y_expand = abs(calibration.y_end_val - calibration.y_start_val) / 2.0
    display_y_min_val = y_low_val - y_expand
    display_y_max_val = y_high_val + y_expand

    display_y_pixels = [
        calibration.real_to_pixel_y(display_y_min_val),
        calibration.real_to_pixel_y(display_y_max_val),
    ]
    crop_y_min = min(display_y_pixels)
    crop_y_max = max(display_y_pixels)

    margin_x = max(8.0, (crop_x_max - crop_x_min) * 0.03)
    margin_y = max(8.0, (crop_y_max - crop_y_min) * 0.06)

    view_x_min = crop_x_min - margin_x
    view_x_max = crop_x_max + margin_x
    view_y_min = crop_y_min - margin_y
    view_y_max = crop_y_max + margin_y
    return view_x_min, view_x_max, view_y_min, view_y_max


def crop_image_with_real_extent(
    image: np.ndarray,
    calibration: Calibration,
    view_x_min: float,
    view_x_max: float,
    view_y_min: float,
    view_y_max: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    image_height, image_width = image.shape[:2]

    x0 = max(0, int(np.floor(min(view_x_min, view_x_max))))
    x1 = min(image_width, int(np.ceil(max(view_x_min, view_x_max))))
    y0 = max(0, int(np.floor(min(view_y_min, view_y_max))))
    y1 = min(image_height, int(np.ceil(max(view_y_min, view_y_max))))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("裁剪区域无效，请检查标定点。")

    cropped_image = image[y0:y1, x0:x1]

    real_x_0 = calibration.pixel_to_real_x(x0)
    real_x_1 = calibration.pixel_to_real_x(x1)
    real_y_0 = calibration.pixel_to_real_y(y0)
    real_y_1 = calibration.pixel_to_real_y(y1)

    extent = (
        min(real_x_0, real_x_1),
        max(real_x_0, real_x_1),
        min(real_y_0, real_y_1),
        max(real_y_0, real_y_1),
    )
    box_aspect = cropped_image.shape[0] / cropped_image.shape[1]
    return cropped_image, extent, box_aspect


def smooth_curve(x_values: pd.Series, y_values: pd.Series, sample_count: int = 800) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = ~(x_values.isna() | y_values.isna())
    x_valid = x_values[valid_mask].to_numpy(dtype=float)
    y_valid = y_values[valid_mask].to_numpy(dtype=float)

    if len(x_valid) <= 1:
        return x_valid, y_valid

    x_dense = np.linspace(x_valid.min(), x_valid.max(), sample_count)

    if importlib.util.find_spec("scipy") is not None and len(x_valid) >= 3:
        from scipy.interpolate import PchipInterpolator

        interpolator = PchipInterpolator(x_valid, y_valid)
        y_dense = interpolator(x_dense)
    else:
        y_dense = np.interp(x_dense, x_valid, y_valid)

    return x_dense, y_dense


def plot_comparison(
    data_frame: pd.DataFrame,
    cropped_image: np.ndarray,
    image_extent: tuple[float, float, float, float],
    box_aspect: float,
    output_path: Path,
    show_figure: bool,
) -> None:
    x_values = data_frame["x"]
    y_min = image_extent[2]
    y_max = image_extent[3]
    blue_x, blue_y = smooth_curve(x_values, data_frame["blue"])
    red_x, red_y = smooth_curve(x_values, data_frame["red"])

    figure, (axis_top, axis_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )

    axis_top.scatter(
        x_values,
        data_frame["black"],
        color="black",
        marker="o",
        s=60,
        facecolors="none",
        linewidths=1.6,
        label="black",
    )
    axis_top.plot(blue_x, blue_y, color="blue", linewidth=2.0, label="blue")
    axis_top.plot(red_x, red_y, color="red", linewidth=2.0, label="red")
    axis_top.set_title("Excel 读取数据")
    axis_top.set_ylabel("y")
    axis_top.set_ylim(y_min, y_max)
    axis_top.set_box_aspect(box_aspect)
    axis_top.grid(True, alpha=0.25)
    axis_top.legend(loc="best")

    axis_bottom.imshow(
        cropped_image,
        extent=image_extent,
        origin="upper",
        aspect="auto",
    )
    axis_bottom.set_title("按步骤 2 视野裁剪的原图")
    axis_bottom.set_xlabel("x")
    axis_bottom.set_ylabel("y")
    axis_bottom.set_ylim(y_min, y_max)
    axis_bottom.set_box_aspect(box_aspect)

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"对比图已保存：{output_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    excel_path = resolve_path("图片1_提取结果.xlsx", 1)
    dat_path = resolve_path("图片1_提取结果.dat", 2)
    image_path = resolve_path("图片1.png", 3)
    output_path = resolve_path("图片1_提取结果对比.png", 4)
    show_figure = "--show" in sys.argv

    calibration = load_calibration_from_dat(dat_path)
    if calibration is None:
        raise FileNotFoundError(f"无法从 DAT 中读取标定信息：{dat_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"原图不存在：{image_path}")

    data_frame = load_excel_data(excel_path)
    image = plt.imread(image_path)
    view_x_min, view_x_max, view_y_min, view_y_max = compute_step2_view(calibration)
    cropped_image, image_extent, box_aspect = crop_image_with_real_extent(
        image=image,
        calibration=calibration,
        view_x_min=view_x_min,
        view_x_max=view_x_max,
        view_y_min=view_y_min,
        view_y_max=view_y_max,
    )
    plot_comparison(
        data_frame=data_frame,
        cropped_image=cropped_image,
        image_extent=image_extent,
        box_aspect=box_aspect,
        output_path=output_path,
        show_figure=show_figure,
    )


if __name__ == "__main__":
    main()

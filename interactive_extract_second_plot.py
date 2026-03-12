from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NUM_POINTS = 40
RIGHT_CLICK_INTERVAL_MS = 500


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    @property
    def encoding(self) -> str:
        for stream in self.streams:
            if getattr(stream, "encoding", None):
                return stream.encoding
        return "utf-8"


def input_float(message: str) -> float:
    while True:
        try:
            return float(input(message).strip())
        except ValueError:
            print("输入无效，请输入数字。")


def resolve_image_path() -> Path:
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1]).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        return image_path

    cwd = Path.cwd()
    preferred = cwd / "图片1.png"
    if preferred.exists():
        return preferred.resolve()

    image_files = sorted(
        path for path in cwd.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError("当前目录下未找到可处理的图片文件。")
    return image_files[0].resolve()


def click_data(fig: plt.Figure, ax: plt.Axes, title: str) -> tuple[float, float]:
    ax.set_title(title)
    fig.canvas.draw()
    plt.pause(0.01)

    points = plt.ginput(1, timeout=-1)
    if not points:
        raise RuntimeError("未检测到点击，流程终止。")

    x_pos, y_pos = points[0]
    ax.plot([x_pos], [y_pos], marker="o", color="cyan", markersize=8)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
    return float(x_pos), float(y_pos)


@dataclass
class Calibration:
    x_start_px: tuple[float, float]
    x_end_px: tuple[float, float]
    y_start_px: tuple[float, float]
    y_end_px: tuple[float, float]
    x_start_val: float
    x_end_val: float
    y_start_val: float
    y_end_val: float

    def validate(self) -> None:
        if abs(self.x_end_px[0] - self.x_start_px[0]) < 1e-9:
            raise ValueError("X 轴标定失败：起点和终点像素横坐标相同。")
        if abs(self.y_end_px[1] - self.y_start_px[1]) < 1e-9:
            raise ValueError("Y 轴标定失败：起点和终点像素纵坐标相同。")
        if abs(self.x_end_val - self.x_start_val) < 1e-9:
            raise ValueError("X 轴标定失败：起点和终点真实值相同。")
        if abs(self.y_end_val - self.y_start_val) < 1e-9:
            raise ValueError("Y 轴标定失败：起点和终点真实值相同。")

    def pixel_to_real_x(self, pixel_x: float) -> float:
        return self.x_start_val + (pixel_x - self.x_start_px[0]) * (
            self.x_end_val - self.x_start_val
        ) / (self.x_end_px[0] - self.x_start_px[0])

    def pixel_to_real_y(self, pixel_y: float) -> float:
        return self.y_start_val + (pixel_y - self.y_start_px[1]) * (
            self.y_end_val - self.y_start_val
        ) / (self.y_end_px[1] - self.y_start_px[1])

    def real_to_pixel_y(self, real_y: float) -> float:
        return self.y_start_px[1] + (real_y - self.y_start_val) * (
            self.y_end_px[1] - self.y_start_px[1]
        ) / (self.y_end_val - self.y_start_val)


def collect_calibration(image: np.ndarray, image_name: str) -> Calibration:
    figure, axis = plt.subplots(figsize=(14, 10))
    axis.imshow(image)
    axis.set_xlabel("请按提示依次点击 4 个标定点")
    axis.set_ylabel("像素坐标")
    plt.tight_layout()
    plt.ion()

    print("=" * 72)
    print(f"当前图片：{image_name}")
    print("步骤 1/2：请依次点击 X 轴起点、X 轴终点、Y 轴起点、Y 轴终点。")
    print("=" * 72)

    x_start_px = click_data(figure, axis, "第 1 步：点击 X 轴起点")
    x_end_px = click_data(figure, axis, "第 2 步：点击 X 轴终点")
    y_start_px = click_data(figure, axis, "第 3 步：点击 Y 轴起点")
    y_end_px = click_data(figure, axis, "第 4 步：点击 Y 轴终点")

    axis.set_title("标定点已选完，请到控制台输入真实值")
    figure.canvas.draw()
    plt.pause(0.01)

    print("\n请输入 4 个标定点对应的真实值：")
    x_start_val = input_float("X 轴起点真实值：")
    x_end_val = input_float("X 轴终点真实值：")
    y_start_val = input_float("Y 轴起点真实值：")
    y_end_val = input_float("Y 轴终点真实值：")

    plt.close(figure)
    plt.ioff()

    calibration = Calibration(
        x_start_px=x_start_px,
        x_end_px=x_end_px,
        y_start_px=y_start_px,
        y_end_px=y_end_px,
        x_start_val=x_start_val,
        x_end_val=x_end_val,
        y_start_val=y_start_val,
        y_end_val=y_end_val,
    )
    calibration.validate()
    return calibration


class CurveCollector:
    def __init__(self, image: np.ndarray, calibration: Calibration, num_points: int = NUM_POINTS) -> None:
        self.image = image
        self.calibration = calibration
        self.num_points = num_points

        self.curves = [
            {
                "key": "black",
                "label": "黑色实验数据",
                "color": "black",
                "sample_indices": list(range(2, self.num_points - 2)),
            },
            {
                "key": "blue",
                "label": "蓝色曲线",
                "color": "blue",
                "sample_indices": list(range(self.num_points)),
            },
            {
                "key": "red",
                "label": "红色曲线",
                "color": "red",
                "sample_indices": list(range(self.num_points)),
            },
        ]
        self.current_curve_index = 0
        self.finished = False

        self.sample_x_pixels = np.linspace(
            self.calibration.x_start_px[0],
            self.calibration.x_end_px[0],
            self.num_points,
        )
        self.sample_x_values = np.linspace(
            self.calibration.x_start_val,
            self.calibration.x_end_val,
            self.num_points,
        )

        self.crop_x_min = min(self.calibration.x_start_px[0], self.calibration.x_end_px[0])
        self.crop_x_max = max(self.calibration.x_start_px[0], self.calibration.x_end_px[0])

        y_low_val = min(self.calibration.y_start_val, self.calibration.y_end_val)
        y_high_val = max(self.calibration.y_start_val, self.calibration.y_end_val)
        y_expand = abs(self.calibration.y_end_val - self.calibration.y_start_val) / 2.0
        display_y_min_val = y_low_val - y_expand
        display_y_max_val = y_high_val + y_expand
        display_y_pixels = [
            self.calibration.real_to_pixel_y(display_y_min_val),
            self.calibration.real_to_pixel_y(display_y_max_val),
        ]
        self.crop_y_min = min(display_y_pixels)
        self.crop_y_max = max(display_y_pixels)

        margin_x = max(8.0, (self.crop_x_max - self.crop_x_min) * 0.03)
        margin_y = max(8.0, (self.crop_y_max - self.crop_y_min) * 0.06)
        self.view_x_min = self.crop_x_min - margin_x
        self.view_x_max = self.crop_x_max + margin_x
        self.view_y_min = self.crop_y_min - margin_y
        self.view_y_max = self.crop_y_max + margin_y

        self.points_px: dict[str, list[tuple[float, float]]] = {curve["key"]: [] for curve in self.curves}
        self.point_artists: dict[str, list[plt.Artist]] = {curve["key"]: [] for curve in self.curves}

        self.figure: plt.Figure | None = None
        self.axis: plt.Axes | None = None
        self.info_text = None
        self.current_line = None
        self.right_click_timer = None
        self.pending_right_click = False

    @property
    def current_curve(self) -> dict[str, object]:
        return self.curves[self.current_curve_index]

    @property
    def current_curve_key(self) -> str:
        return self.current_curve["key"]

    @property
    def current_curve_sample_indices(self) -> list[int]:
        return self.current_curve["sample_indices"]

    @property
    def current_curve_target_count(self) -> int:
        return len(self.current_curve_sample_indices)

    @property
    def current_curve_points(self) -> list[tuple[float, float]]:
        return self.points_px[self.current_curve_key]

    def setup_plot(self) -> None:
        self.figure, self.axis = plt.subplots(figsize=(15, 8))
        self.axis.imshow(self.image)
        self.axis.set_xlim(self.view_x_min, self.view_x_max)
        self.axis.set_ylim(self.view_y_max, self.view_y_min)
        self.axis.set_xlabel("x 像素坐标（已裁剪放大显示）")
        self.axis.set_ylabel("y 像素坐标")

        for x_pixel in self.sample_x_pixels:
            self.axis.axvline(
                x=x_pixel,
                color="0.7",
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )

        label_y = self.crop_y_min + (self.crop_y_max - self.crop_y_min) * 0.04
        for index, x_pixel in enumerate(self.sample_x_pixels, start=1):
            self.axis.text(
                x_pixel,
                label_y,
                str(index),
                color="0.45",
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=2,
            )

        self.current_line = self.axis.axvline(
            x=self.sample_x_pixels[self.current_curve_sample_indices[0]],
            color="orange",
            linewidth=2.0,
            alpha=0.95,
            zorder=3,
        )

        self.info_text = self.figure.text(0.02, 0.02, "", fontsize=11)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.update_status(
            "左键记录当前垂线交点；右键一次撤销；连续右击两次切换下一条曲线。"
        )
        self.figure.tight_layout(rect=(0, 0.05, 1, 0.96))

    def update_status(self, message: str = "") -> None:
        if self.axis is None or self.info_text is None:
            return

        current_count = len(self.current_curve_points)
        target_count = self.current_curve_target_count
        if current_count < target_count:
            sample_index = self.current_curve_sample_indices[current_count]
            x_value = self.sample_x_values[sample_index]
            title = (
                f"当前：{self.current_curve['label']} | "
                f"点 {current_count + 1}/{target_count} | "
                f"垂线 {sample_index + 1}/{self.num_points} | x = {x_value:.6g}"
            )
        else:
            title = (
                f"当前：{self.current_curve['label']} | "
                f"已完成 {target_count}/{target_count}"
            )

        self.axis.set_title(title)
        helper = "顺序：黑色实验数据 → 蓝色曲线 → 红色曲线"
        self.info_text.set_text(f"{helper}\n{message}".strip())
        self.refresh_current_line()

    def refresh_current_line(self) -> None:
        if self.figure is None or self.current_line is None:
            return

        current_count = len(self.current_curve_points)
        target_count = self.current_curve_target_count
        if current_count < target_count:
            sample_index = self.current_curve_sample_indices[current_count]
            x_pixel = self.sample_x_pixels[sample_index]
            self.current_line.set_xdata([x_pixel, x_pixel])
            self.current_line.set_visible(True)
        else:
            self.current_line.set_visible(False)

        self.figure.canvas.draw_idle()

    def on_click(self, event) -> None:
        if self.finished or self.axis is None or event.inaxes != self.axis:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == MouseButton.LEFT:
            self.record_point(float(event.ydata))
        elif event.button == MouseButton.RIGHT:
            self.handle_right_click()

    def handle_right_click(self) -> None:
        if self.figure is None:
            return

        if self.pending_right_click:
            if self.right_click_timer is not None:
                self.right_click_timer.stop()
            self.pending_right_click = False
            self.advance_curve()
            return

        self.pending_right_click = True
        self.right_click_timer = self.figure.canvas.new_timer(interval=RIGHT_CLICK_INTERVAL_MS)
        if hasattr(self.right_click_timer, "single_shot"):
            self.right_click_timer.single_shot = True
        self.right_click_timer.add_callback(self.commit_single_right_click)
        self.right_click_timer.start()

    def commit_single_right_click(self) -> None:
        self.pending_right_click = False
        self.undo_last_point()

    def record_point(self, pixel_y: float) -> None:
        current_points = self.current_curve_points
        target_count = self.current_curve_target_count
        if len(current_points) >= target_count:
            self.update_status("本条曲线已经采满，请连续右击两次进入下一条曲线。")
            return

        sample_index = self.current_curve_sample_indices[len(current_points)]
        pixel_x = float(self.sample_x_pixels[sample_index])
        pixel_y = float(np.clip(pixel_y, self.crop_y_min, self.crop_y_max))
        current_points.append((pixel_x, pixel_y))

        artist = self.axis.scatter(
            [pixel_x],
            [pixel_y],
            s=54,
            facecolors="none",
            edgecolors=self.current_curve["color"],
            linewidths=1.8,
            zorder=4,
        )
        self.point_artists[self.current_curve_key].append(artist)

        point_index = len(current_points)
        real_x = self.sample_x_values[sample_index]
        real_y = self.calibration.pixel_to_real_y(pixel_y)
        print(
            f"[{self.current_curve['label']}] 第 {point_index:02d}/{target_count} 个点"
            f"（垂线 {sample_index + 1}/{self.num_points}）："
            f"x = {real_x:.6f}, y = {real_y:.6f}"
        )

        if point_index == target_count:
            self.update_status("本条曲线已完成。右键一次可撤销最后一点；连续右击两次进入下一条曲线。")
        else:
            self.update_status(f"已记录第 {point_index} 个点。")

    def undo_last_point(self) -> None:
        current_points = self.current_curve_points
        if not current_points:
            self.update_status("当前曲线还没有可撤销的点。")
            return

        removed_point = current_points.pop()
        artist = self.point_artists[self.current_curve_key].pop()
        artist.remove()

        next_index = len(current_points) + 1
        print(
            f"[{self.current_curve['label']}] 已撤销上一点，回到第 {next_index:02d}/{self.current_curve_target_count} 个点。"
            f" 撤销点像素坐标 = ({removed_point[0]:.2f}, {removed_point[1]:.2f})"
        )
        self.update_status("已撤销上一点。")

    def advance_curve(self) -> None:
        current_count = len(self.current_curve_points)
        target_count = self.current_curve_target_count
        if current_count != target_count:
            remaining = target_count - current_count
            self.update_status(f"当前曲线还差 {remaining} 个点，不能切换。")
            return

        if self.current_curve_index == len(self.curves) - 1:
            self.finished = True
            self.update_status("三条曲线都已完成，窗口将关闭并保存结果。")
            plt.pause(0.2)
            plt.close(self.figure)
            return

        self.current_curve_index += 1
        print(f"\n>>> 进入下一条曲线：{self.current_curve['label']}")
        self.update_status("请继续沿当前高亮垂线点击对应曲线交点。")

    def run(self) -> bool:
        self.setup_plot()

        print("\n步骤 2/2：开始采点。")
        print("采点顺序固定为：黑色实验数据 → 蓝色曲线 → 红色曲线。")
        print("黑色实验数据只采中间 36 个点，对应第 3 到第 38 条垂线；蓝线和红线仍采 40 个点。")
        print("左键：记录当前高亮垂线上的交点。")
        print("右键一次：撤销当前曲线的上一个点。")
        print("连续右击两次（间隔约 0.5 秒内）：当且仅当当前曲线已采满时，切换到下一条曲线。")
        print("第三条曲线完成后，再连续右击两次即可结束并保存。")
        print()

        plt.show()
        return self.finished

    def build_output_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        aligned_y_values: dict[str, list[float]] = {
            curve["key"]: [np.nan] * self.num_points for curve in self.curves
        }

        for curve in self.curves:
            curve_key = curve["key"]
            for sample_index, (_, pixel_y) in zip(curve["sample_indices"], self.points_px[curve_key]):
                aligned_y_values[curve_key][sample_index] = self.calibration.pixel_to_real_y(pixel_y)

        data_frame = pd.DataFrame(
            {
                "black": aligned_y_values["black"],
                "blue": aligned_y_values["blue"],
                "red": aligned_y_values["red"],
            }
        )
        x_frame = pd.DataFrame({"x": self.sample_x_values})
        return data_frame, x_frame


def export_to_excel(output_path: Path, data_frame: pd.DataFrame, x_frame: pd.DataFrame) -> None:
    engine = None
    if importlib.util.find_spec("openpyxl") is not None:
        engine = "openpyxl"
    elif importlib.util.find_spec("xlsxwriter") is not None:
        engine = "xlsxwriter"

    if engine is None:
        raise ModuleNotFoundError("未检测到 `openpyxl` 或 `xlsxwriter`，无法导出 Excel。")

    combined_frame = pd.concat([x_frame, data_frame], axis=1)

    with pd.ExcelWriter(output_path, engine=engine) as writer:
        combined_frame.to_excel(writer, sheet_name="data", index=False)
        x_frame.to_excel(writer, sheet_name="x_values", index=False)


def export_to_dat(
    output_path: Path,
    calibration: Calibration,
    x_frame: pd.DataFrame | None = None,
    data_frame: pd.DataFrame | None = None,
) -> None:
    def format_number(value: float) -> str:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.6f}"

    lines = [
        "# calibration_points",
        "# label\tpixel_x\tpixel_y\taxis_value",
        f"x_start\t{calibration.x_start_px[0]:.6f}\t{calibration.x_start_px[1]:.6f}\t{calibration.x_start_val:.6f}",
        f"x_end\t{calibration.x_end_px[0]:.6f}\t{calibration.x_end_px[1]:.6f}\t{calibration.x_end_val:.6f}",
        f"y_start\t{calibration.y_start_px[0]:.6f}\t{calibration.y_start_px[1]:.6f}\t{calibration.y_start_val:.6f}",
        f"y_end\t{calibration.y_end_px[0]:.6f}\t{calibration.y_end_px[1]:.6f}\t{calibration.y_end_val:.6f}",
    ]

    if x_frame is not None and data_frame is not None:
        lines.extend(
            [
                "",
                "# extracted_data",
                "# index\tx\tblack\tblue\tred",
            ]
        )

        for row_index in range(len(x_frame)):
            lines.append(
                "\t".join(
                    [
                        str(row_index + 1),
                        format_number(x_frame.iloc[row_index, 0]),
                        format_number(data_frame.iloc[row_index, 0]),
                        format_number(data_frame.iloc[row_index, 1]),
                        format_number(data_frame.iloc[row_index, 2]),
                    ]
                )
            )

    with output_path.open("w", encoding="utf-8", newline="\n") as dat_file:
        dat_file.write("\n".join(lines) + "\n")


def load_calibration_from_dat(dat_path: Path) -> Calibration | None:
    if not dat_path.exists():
        return None

    entries: dict[str, tuple[float, float, float]] = {}
    valid_labels = {"x_start", "x_end", "y_start", "y_end"}

    try:
        with dat_path.open("r", encoding="utf-8") as dat_file:
            for raw_line in dat_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) != 4 or parts[0] not in valid_labels:
                    continue

                label = parts[0]
                pixel_x = float(parts[1])
                pixel_y = float(parts[2])
                axis_value = float(parts[3])
                entries[label] = (pixel_x, pixel_y, axis_value)
    except (OSError, ValueError):
        return None

    if set(entries) != valid_labels:
        return None

    calibration = Calibration(
        x_start_px=(entries["x_start"][0], entries["x_start"][1]),
        x_end_px=(entries["x_end"][0], entries["x_end"][1]),
        y_start_px=(entries["y_start"][0], entries["y_start"][1]),
        y_end_px=(entries["y_end"][0], entries["y_end"][1]),
        x_start_val=entries["x_start"][2],
        x_end_val=entries["x_end"][2],
        y_start_val=entries["y_start"][2],
        y_end_val=entries["y_end"][2],
    )

    try:
        calibration.validate()
    except ValueError:
        return None

    return calibration


def run_interactive_extraction() -> None:
    image_path = resolve_image_path()
    image = plt.imread(image_path)
    dat_output_path = image_path.with_name(f"{image_path.stem}_提取结果.dat")

    calibration = load_calibration_from_dat(dat_output_path)
    if calibration is None:
        calibration = collect_calibration(image, image_path.name)
        export_to_dat(dat_output_path, calibration)
        print(f"已保存坐标轴标定信息：{dat_output_path}")
    else:
        print("=" * 72)
        print(f"检测到已有标定文件，已自动加载：{dat_output_path}")
        print("本次跳过坐标轴参考点选择，直接进入曲线提取。")
        print("=" * 72)

    collector = CurveCollector(image=image, calibration=calibration, num_points=NUM_POINTS)
    completed = collector.run()

    if not completed:
        print("窗口在完成全部采点前被关闭，未导出 Excel。")
        return

    data_frame, x_frame = collector.build_output_frames()
    output_path = image_path.with_name(f"{image_path.stem}_提取结果.xlsx")
    export_to_excel(output_path, data_frame, x_frame)
    export_to_dat(dat_output_path, calibration, x_frame, data_frame)

    print("=" * 72)
    print("提取完成。")
    print(f"Excel 已保存：{output_path}")
    print(f"DAT 已保存：{dat_output_path}")
    print("data 工作表：第一列为 x，后 3 列为 black / blue / red。")
    print("x_values 工作表：保存这 40 个均匀采样点对应的 x 值。")
    print("=" * 72)


def main() -> None:
    log_path = Path.cwd() / "logs.txt"
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"日志输出已自动记录到：{log_path}")
            run_interactive_extraction()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()

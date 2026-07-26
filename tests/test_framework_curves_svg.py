from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "paper" / "plot_framework_curves.py"


def _load_generator():
    assert GENERATOR_PATH.is_file(), "framework curve generator is missing"
    spec = spec_from_file_location("plot_framework_curves", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_renders_editable_failure_and_success_svgs(tmp_path):
    generator = _load_generator()
    failure, failure_spread, success, success_spread = generator.build_curve_data()

    assert failure[0] > failure[-1]
    assert success[0] < success[-1]

    cases = (
        ("failure", failure, failure_spread, "#F6C1C8"),
        ("success", success, success_spread, "#46A253"),
    )
    for name, mean, spread, color in cases:
        output_path = tmp_path / f"{name}.svg"
        generator.render_curve(output_path, mean, spread, color)

        ElementTree.parse(output_path)
        svg = output_path.read_text(encoding="utf-8")
        assert "<text" in svg
        assert "Env steps (k)" in svg
        assert "Eval success rate" in svg
        assert color.lower() in svg
        assert "fill-opacity: 0.2" in svg
        assert all(line == line.rstrip() for line in svg.splitlines())


def test_plot_content_has_safe_canvas_margins(tmp_path):
    generator = _load_generator()
    failure, failure_spread, _, _ = generator.build_curve_data()
    output_path = tmp_path / "failure.png"
    generator.render_curve(output_path, failure, failure_spread, "#F6C1C8")

    with Image.open(output_path) as image:
        rgb = image.convert("RGB")
        difference = ImageChops.difference(
            rgb,
            Image.new("RGB", rgb.size, "white"),
        )
        content_bbox = difference.getbbox()

    assert content_bbox is not None
    left, top, right, bottom = content_bbox
    width, height = rgb.size
    assert left >= 2
    assert top >= 2
    assert right <= width - 2
    assert bottom <= height - 2

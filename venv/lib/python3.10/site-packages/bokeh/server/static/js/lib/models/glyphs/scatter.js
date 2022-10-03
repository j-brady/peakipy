var _a;
import { Marker, MarkerView } from "./marker";
import { marker_funcs } from "./defs";
import * as p from "../../core/properties";
export class ScatterView extends MarkerView {
    async lazy_initialize() {
        await super.lazy_initialize();
        const { webgl } = this.renderer.plot_view.canvas_view;
        if (webgl != null && webgl.regl_wrapper.has_webgl) {
            const { MultiMarkerGL } = await import("./webgl/multi_marker");
            this.glglyph = new MultiMarkerGL(webgl.regl_wrapper, this);
        }
    }
    _render(ctx, indices, data) {
        const { sx, sy, size, angle, marker } = data ?? this;
        for (const i of indices) {
            const sx_i = sx[i];
            const sy_i = sy[i];
            const size_i = size.get(i);
            const angle_i = angle.get(i);
            const marker_i = marker.get(i);
            if (!isFinite(sx_i + sy_i + size_i + angle_i) || marker_i == null)
                continue;
            const r = size_i / 2;
            ctx.beginPath();
            ctx.translate(sx_i, sy_i);
            if (angle_i)
                ctx.rotate(angle_i);
            marker_funcs[marker_i](ctx, i, r, this.visuals);
            if (angle_i)
                ctx.rotate(-angle_i);
            ctx.translate(-sx_i, -sy_i);
        }
    }
    draw_legend_for_index(ctx, { x0, x1, y0, y1 }, index) {
        const n = index + 1;
        const marker = this.marker.get(index);
        const args = {
            ...this._get_legend_args({ x0, x1, y0, y1 }, index),
            marker: new p.UniformScalar(marker, n),
        };
        this._render(ctx, [index], args);
    }
}
ScatterView.__name__ = "ScatterView";
export class Scatter extends Marker {
    constructor(attrs) {
        super(attrs);
    }
}
_a = Scatter;
Scatter.__name__ = "Scatter";
(() => {
    _a.prototype.default_view = ScatterView;
    _a.define(() => ({
        marker: [p.MarkerSpec, { value: "circle" }],
    }));
})();
//# sourceMappingURL=scatter.js.map